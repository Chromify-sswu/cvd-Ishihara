# ishihara.py
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import color
from scipy.optimize import curve_fit

# — LAB <-> RGB 변환 —
def lab_to_rgb(L, a, b):
    """Convert LAB to RGB with proper clamping"""
    lab = np.array([[[L, a, b]]], dtype=np.float64)
    rgb = color.lab2rgb(lab)
    rgb8 = tuple((np.clip(rgb[0, 0], 0, 1) * 255).astype(np.uint8))
    return rgb8


def generate_ishihara_plate(number, axis, deltaE, rng, size=600):
    # 축별 특성 정의
    if axis == 'protan':
        # 적색맹: 어두운 적-녹 계열, a* 축 차이
        L_base_mean = 50.0  # 어둡게
        base_a_center = rng.uniform(0, 20)
        base_b_center = rng.uniform(5, 15)
        relative_a_diff = deltaE * rng.uniform(0.9, 1.1)
        relative_b_diff = 0

    elif axis == 'deutan':
        # 녹색맹: 밝은 적-녹 계열, a* 축 차이
        L_base_mean = 70.0  # 밝게
        base_a_center = rng.uniform(-10, 10)
        base_b_center = rng.uniform(10, 25)
        relative_a_diff = deltaE * rng.uniform(0.9, 1.1)
        relative_b_diff = 0

    elif axis == 'tritan':
        # 청색맹: 청-황 계열, b* 축 차이
        L_base_mean = 60.0
        base_a_center = rng.uniform(-5, 5)
        base_b_center = rng.uniform(-15, 5)
        relative_a_diff = 0
        relative_b_diff = deltaE * rng.uniform(0.9, 1.1)

    else:  # 기본값 (Fallback)
        L_base_mean = 60.0
        base_a_center = 10
        base_b_center = 15
        relative_a_diff = deltaE * 0.9
        relative_b_diff = 0

    bg_lab_base = (L_base_mean,
                   base_a_center - relative_a_diff / 2,
                   base_b_center - relative_b_diff / 2)
    fg_lab_base = (L_base_mean,
                   base_a_center + relative_a_diff / 2,
                   base_b_center + relative_b_diff / 2)

    img = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(img)

    mask = Image.new('L', (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)

    font_size = int(size * 0.65)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                font_size
            )
        except Exception:
            font = ImageFont.load_default()

    text = str(number)
    bbox = mask_draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = (size - text_w) // 2 - bbox[0]
    text_y = (size - text_h) // 2 - bbox[1]
    mask_draw.text((text_x, text_y), text, font=font, fill=255)

    mask_array = np.array(mask)

    center_x, center_y = size // 2, size // 2
    radius = int(size * 0.45)

    num_circles = 6000
    circles_to_draw = []

    for _ in range(num_circles):
        angle = rng.random() * 2 * math.pi
        r_dist = math.sqrt(rng.random())
        r = r_dist * radius
        x = int(center_x + r * math.cos(angle))
        y = int(center_y + r * math.sin(angle))

        # 크기 분포
        rand_val = rng.random()
        if rand_val < 0.6:
            circle_radius = rng.integers(2, 4)
        elif rand_val < 0.9:
            circle_radius = rng.integers(5, 8)
        else:
            circle_radius = rng.integers(9, 12)

        if 0 <= x < size and 0 <= y < size:
            in_number = mask_array[y, x] > 128

            L_variation = 25.0
            random_L = rng.uniform(L_base_mean - L_variation,
                                   L_base_mean + L_variation)
            random_L = np.clip(random_L, 0, 100)

            current_fg_lab = (random_L, fg_lab_base[1], fg_lab_base[2])
            current_bg_lab = (random_L, bg_lab_base[1], bg_lab_base[2])

            if in_number:
                contamination = 0.05
                final_base_lab = (current_fg_lab if rng.random() > contamination
                                  else current_bg_lab)
                L_noise = rng.uniform(-8, 8)
                a_noise = rng.uniform(-20, 20)
                b_noise = rng.uniform(-20, 20)
            else:
                contamination = 0.08
                final_base_lab = (current_bg_lab if rng.random() > contamination
                                  else current_fg_lab)
                L_noise = rng.uniform(-18, 18)
                a_noise = rng.uniform(-40, 40)
                b_noise = rng.uniform(-40, 40)

            final_lab = (
                np.clip(final_base_lab[0] + L_noise, 0, 100),
                final_base_lab[1] + a_noise,
                final_base_lab[2] + b_noise
            )

            final_rgb = lab_to_rgb(*final_lab)

            r_val, g_val, b_val = [
                int(np.clip(int(c) + rng.integers(-3, 4), 0, 255))
                for c in final_rgb
            ]
            circles_to_draw.append((x, y, circle_radius,
                                    (r_val, g_val, b_val)))

    for x, y, r, col in circles_to_draw:
        bbox = [x - r, y - r, x + r, y + r]
        draw.ellipse(bbox, fill=col)

    return img


class AdaptiveStaircase:
    """
    2-down-1-up 규칙
    'Reversal(반전)' 기반으로 step_size를 조절 (Coarse-to-Fine)
    """
    def __init__(self, deltas, start_index=None):
        self.deltas = sorted(deltas, reverse=True)
        self.index = (start_index
                      if start_index is not None
                      else len(self.deltas) // 3)
        self.history = []
        self.consecutive_correct = 0

        self.step_size_large = 3
        self.step_size_small = 1
        self.step_size = self.step_size_large
        self.reversals = 0
        self.last_direction = None

    def current_delta(self):
        return self.deltas[self.index]

    def record(self, correct):
        self.history.append((self.current_delta(), int(correct)))
        current_direction = None

        if correct:
            self.consecutive_correct += 1
            if self.consecutive_correct >= 2:
                self.consecutive_correct = 0
                # 2번 맞힘: 어렵게 (index 증가)
                new_index = min(len(self.deltas) - 1,
                                self.index + self.step_size)
                if new_index != self.index:
                    self.index = new_index
                    current_direction = 'down'
        else:
            self.consecutive_correct = 0
            # 1번 틀림: 쉽게 (index 감소)
            new_index = max(0, self.index - self.step_size)
            if new_index != self.index:
                self.index = new_index
                current_direction = 'up'

        if current_direction and self.last_direction:
            if current_direction != self.last_direction:
                self.reversals += 1
                if self.reversals >= 2:
                    self.step_size = self.step_size_small

        if current_direction:
            self.last_direction = current_direction


def calculate_threshold_and_confidence(stair_history):
    """
    Psychometric + reversal 기반 역치 추정
    Returns: (threshold: float or None, confidence: float 0..1)
    """
    if len(stair_history) < 10:
        return None, 0.0

    # — 1) 반전점 탐지 (초기 2개 반전 무시) —
    up_reversals = []
    down_reversals = []
    last_direction = None
    consecutive_correct = 0
    reversals_found = 0

    for i in range(len(stair_history)):
        delta, correct = stair_history[i]
        current_dir = None

        if correct:
            consecutive_correct += 1
            if consecutive_correct >= 2:  # 2-down
                consecutive_correct = 0
                current_dir = 'down'
        else:
            consecutive_correct = 0
            current_dir = 'up'  # 1-up

        if current_dir and last_direction and current_dir != last_direction:
            reversals_found += 1
            if reversals_found > 2:
                if current_dir == 'up':
                    down_reversals.append(delta)
                else:
                    up_reversals.append(delta)

        if current_dir:
            last_direction = current_dir

    # — 2) 반전 평균 기반 역치 후보 —
    if up_reversals and down_reversals:
        reversal_mean = (np.mean(up_reversals)
                         + np.mean(down_reversals)) / 2.0
    elif up_reversals:
        reversal_mean = float(np.mean(up_reversals))
    elif down_reversals:
        reversal_mean = float(np.mean(down_reversals))
    else:
        reversal_mean = np.mean([d for d, _ in stair_history[-5:]])
        if not reversal_mean:
            return None, 0.0

    # — 3) Psychometric fitting (로지스틱) —
    def psychometric_func(deltaE, alpha, beta):
        # 2-down-1-up은 ~70.7% 지점을 찾습니다.
        return 1.0 / (1.0 + np.exp(-(deltaE - alpha) / beta))

    deltaE_arr = np.array([d for d, _ in stair_history])
    corrects = np.array([int(c) for _, c in stair_history])

    try:
        popt, _ = curve_fit(
            psychometric_func,
            deltaE_arr,
            corrects,
            p0=[np.mean(deltaE_arr), 5.0],
            bounds=([0.0, 0.1], [100.0, 20.0]),
            maxfev=5000
        )
        alpha, beta = popt
        threshold_model = float(alpha + 0.881 * beta)  # 70.7% 지점

        # model fit confidence (R^2-like)
        y_pred = psychometric_func(deltaE_arr, alpha, beta)
        ss_res = np.sum((corrects - y_pred) ** 2)
        ss_tot = np.sum((corrects - np.mean(corrects)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        confidence = float(np.clip(r2, 0.0, 1.0))

    except Exception:
        threshold_model = float(reversal_mean)
        confidence = 0.25

    # — 4) 결합 (가중 결합: 모델 우선, 반전 보정) —
    final_threshold = float(
        (threshold_model * 0.7) + (reversal_mean * 0.3)
    )

    # — 5) 안정성 보정 (데이터 부족 시 confidence 축소) —
    if confidence < 0.4 and len(stair_history) < 20:
        confidence *= (len(stair_history) / 20.0)

    return round(final_threshold, 2), round(float(np.clip(confidence, 0.0, 1.0)), 2)


def interpret_threshold(thresh, axis_type):
    """역치를 사람이 읽기 쉬운 문장으로 변환 (단순화)"""
    if thresh is None:
        return "측정 불가", "데이터 부족"

    color_name = {
        'protan': '빨간색',
        'deutan': '초록색',
        'tritan': '파란색/노란색'
    }

    if thresh < 20:
        level = "매우 우수"
    elif thresh < 30:
        level = "우수"
    elif thresh < 40:
        level = "보통"
    elif thresh < 50:
        level = "약간 어려움"
    else:
        level = "어려움"

    desc = f"{color_name.get(axis_type, '색상')} 구분에 {level} 수준입니다."

    return level, desc
