import mss
import numpy as np
import cv2
import pyautogui
import time
from collections import deque

# ── 物理常數（依你的遊戲畫面調整）──
GRAVITY_PX_S2 = 900      # 重力加速度估算
JUMP_VELOCITY  = -328    # 按空白後給的初速（負 = 向上），需實測
SCREEN_MID_Y   = 460
SCREEN_TOP     = 60
SCREEN_BOT     = 820
GAP_HALF       = 60      # 缺口上下各容許多少 px

# ── 跳躍控制 ──
JUMP_COOLDOWN      = 0.18   # 兩次跳躍最短間隔
SUPPRESS_AFTER_JUMP = 0.12  # 跳後多久內不再跳（讓鳥先往上）


def jump_action():
    pyautogui.keyDown('space')
    time.sleep(0.01)
    pyautogui.keyUp('space')


# ════════════════════════════════
#  Bird Tracker（速度 + 預測）
# ════════════════════════════════
class BirdTracker:
    def __init__(self):
        self.history = deque(maxlen=8)   # (t, y)

    def update(self, y):
        if y is not None:
            self.history.append((time.time(), y))

    def velocity(self):
        """最近兩幀的瞬時速度 px/s（正 = 向下）"""
        if len(self.history) < 2:
            return 0
        t1, y1 = self.history[-2]
        t2, y2 = self.history[-1]
        dt = t2 - t1
        return (y2 - y1) / dt if dt > 1e-5 else 0

    def predict(self, t_ahead):
        """預測 t 秒後的 y，只做輔助參考用"""
        if not self.history:
            return None
        _, y0 = self.history[-1]
        v = self.velocity()
        return y0 + v * t_ahead + 0.5 * GRAVITY_PX_S2 * t_ahead ** 2


# ════════════════════════════════
#  Pipe Detector
# ════════════════════════════════
def detect_pipe(frame):
    roi = frame[:825, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([90, 255, 255]))
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pipes = [(cv2.boundingRect(c)) for c in contours if cv2.contourArea(c) > 500]
    if len(pipes) < 2:
        return None, None

    pipes.sort(key=lambda p: p[0])

    # 找最近的一對上下管
    for i in range(len(pipes)):
        for j in range(i + 1, len(pipes)):
            p1, p2 = pipes[i], pipes[j]
            if abs(p1[0] - p2[0]) < 60:
                top    = p1 if p1[1] < p2[1] else p2
                bottom = p2 if top is p1 else p1
                gap_y  = (top[1] + top[3] + bottom[1]) // 2
                pipe_x = top[0] + top[2]   # 管子右緣
                return gap_y, pipe_x

    return None, None


def detectBird(frame):
    y1, y2, x1, x2 = 50, 800, 115, 265
    roi = frame[y1:y2, x1:x2]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
    M = cv2.moments(mask)
    if M["m00"] > 50:
        return int(M["m01"] / M["m00"]) + y1
    return None


# ════════════════════════════════
#  核心決策：目標區間 + 抑制
# ════════════════════════════════
class JumpController:
    def __init__(self):
        self.last_jump_time   = 0
        self.suppressed_until = 0   # 跳後鎖定期

    def decide(self, bird_y, velocity, gap_y, pipe_x, bird_x=190):
        now = time.time()

        if bird_y is None:
            return False

        # 1. 算出目標區間
        if gap_y is not None and pipe_x is not None:
            dist = pipe_x - bird_x
            # 越近容許誤差越小，對準越精確
            half = GAP_HALF if dist > 200 else int(GAP_HALF * 0.6)
            target   = gap_y
            zone_top = target - half
            zone_bot = target + half
        else:
            # 沒有管子，維持畫面中央
            zone_top = SCREEN_MID_Y - 60
            zone_bot = SCREEN_MID_Y + 60

        zone_top = max(zone_top, SCREEN_TOP)
        zone_bot = min(zone_bot, SCREEN_BOT)

        # 2. 鳥衝出上緣 → 強制抑制（不跳，讓重力把鳥拉回來）
        if bird_y < zone_top:
            # 延長抑制時間，依上衝程度決定
            overshoot = zone_top - bird_y
            extra = overshoot / 600   # 每超出 600px 多壓 1 秒
            self.suppressed_until = now + 0.05 + extra
            return False

        # 3. 在抑制期內 → 不跳
        if now < self.suppressed_until:
            return False

        # 4. 冷卻期內 → 不跳
        if now - self.last_jump_time < JUMP_COOLDOWN:
            return False

        # 5. 鳥在安全區間內，但速度往下很快 → 提早跳
        #    用「預測落點」判斷會不會掉出下緣
        t_react = 0.10   # 假設反應時間 0.1 秒
        v = velocity
        predicted = bird_y + v * t_react + 0.5 * GRAVITY_PX_S2 * t_react ** 2
        if predicted > zone_bot:
            return True   # 預測會掉出去，現在跳

        # 6. 已經掉出下緣 → 立刻跳
        if bird_y > zone_bot:
            return True

        return False

    def on_jump(self):
        now = time.time()
        self.last_jump_time   = now
        self.suppressed_until = now + SUPPRESS_AFTER_JUMP


# ════════════════════════════════
#  主迴圈
# ════════════════════════════════
def process():
    tracker    = BirdTracker()
    controller = JumpController()

    with mss.mss() as sct:
        monitor = {"top": 140, "left": 135, "height": 935, "width": 690}
        while True:
            img   = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            bird_y = detectBird(frame)
            tracker.update(bird_y)

            gap_y, pipe_x = detect_pipe(frame)
            velocity      = tracker.velocity()

            if controller.decide(bird_y, velocity, gap_y, pipe_x):
                jump_action()
                controller.on_jump()

            # ── Debug 視覺化 ──
            if bird_y:
                cv2.circle(frame, (190, bird_y), 6, (0, 0, 255), -1)
            if gap_y:
                cv2.circle(frame, (350, gap_y), 6, (255, 0, 0), -1)
                cv2.line(frame, (0, gap_y - GAP_HALF), (frame.shape[1], gap_y - GAP_HALF), (0,200,200), 1)
                cv2.line(frame, (0, gap_y + GAP_HALF), (frame.shape[1], gap_y + GAP_HALF), (0,200,200), 1)

            cv2.putText(frame, f"vel: {velocity:+.0f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            cv2.putText(frame, f"bird_y: {bird_y}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            cv2.imshow("debug", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break