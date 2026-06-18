import cv2
import mediapipe as mp
import numpy as np
import warnings
import requests
import threading
import pyttsx3
from queue import Queue
import time
import sys
from typing import Optional, List, Dict

# ===================== CONFIG (LLM + TTS) =====================

USE_OLLAMA = True  # set False to disable LLM feedback
OLLAMA_MODEL = "squatper"  # local Ollama model name
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TIMEOUT = 30  # seconds

USE_TTS = True      # set False to mute spoken feedback
TTS_RATE = 170      # words per minute
TTS_VOLUME = 1.0   # 0.0 - 1.0
TTS_VOICE_INDEX = None  # None = default; or set to an int after listing voices

# LLM Timing: First call on first rep, then every 3 seconds with latest rep data
LLM_INTERVAL_SECONDS = 2  # Interval between LLM calls after the first one

window_name = 'Squat Form Analyzer'

warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

# ===================== BIG FIVE PERSONALITY CONFIG =====================

# Default Big Five personality scores (1–10). You will be prompted to change them.
BIG5_PERSONALITY = {
    "openness": 5.0,
    "conscientiousness": 5.0,
    "extraversion": 5.0,
    "agreeableness": 5.0,
    "neuroticism": 5.0,
}

def format_big5_for_prompt() -> str:
    """Adapt the feedback that would be suitable matching Big Five profile."""
    b = BIG5_PERSONALITY
    return (
        f"- Openness: {b['openness']:.1f} / 10\n"
        f"- Conscientiousness: {b['conscientiousness']:.1f} / 10\n"
        f"- Extraversion: {b['extraversion']:.1f} / 10\n"
        f"- Agreeableness: {b['agreeableness']:.1f} / 10\n"
        f"- Neuroticism: {b['neuroticism']:.1f} / 10\n"
    )


def get_big5_from_user():
    """Ask the user for Big Five scores (1–10) and store in BIG5_PERSONALITY."""
    if not USE_OLLAMA:
        return  # no need if LLM is disabled

    print("\nEnter Big Five personality scores (1–10). Press Enter to keep default 5.0 for each trait.")
    traits_order = [
        ("openness", "Openness"),
        ("conscientiousness", "Conscientiousness"),
        ("extraversion", "Extraversion"),
        ("agreeableness", "Agreeableness"),
        ("neuroticism", "Neuroticism"),
    ]
    for key, label in traits_order:
        while True:
            s = input(f"{label} (1–10, default 5): ").strip()
            if s == "":
                # keep default
                break
            try:
                val = float(s)
                if 1.0 <= val <= 10.0:
                    BIG5_PERSONALITY[key] = val
                    break
                else:
                    print("  Please enter a number between 1 and 10.")
            except ValueError:
                print("  Invalid input, please enter a number between 1 and 10.")

    print("\nUsing Big Five profile:")
    print(format_big5_for_prompt())



# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Color definitions
COLORS = {
    'blue': (255, 127, 0),
    'red': (50, 50, 255),
    'green': (127, 255, 0),
    'light_green': (127, 233, 100),
    'yellow': (0, 255, 255),
    'magenta': (255, 0, 255),
    'white': (255, 255, 255),
    'cyan': (255, 255, 0),
    'light_blue': (255, 200, 100),
    'orange': (0, 165, 255)
}

# Thresholds
THRESHOLDS = {
    'HIP_KNEE_VERT': {
        'NORMAL': [0, 30],
        'TRANS': [30, 50],
        'PASS': [50, 100]
    }
}

ISSUE_HINTS = {
    "Bend forward": "Bend forward",              # same text as issue name
    "Bend backward": "Control leaning",            # same text as issue name
    "Shallow squat": "go deeper",
    "Deep squat": "less deeper",
    "Knee crossing toe": "control knee position",
}
# Global state tracker (will be reset per run)
state_tracker = {
    'state_seq': [],
    'prev_state': None,
    'curr_state': None,
    'SQUAT_COUNT': 0,
    'angles_during_rep': {
        'spine': [],
        'knee': [],
        'ankle': []
    },
    'rep_data': [],
    'current_rep_issues': [],
    'current_rep_score': None,
    'show_feedback': False,
    'total_issues': 0,
    'total_compliance': 0,
    'feedback_start_time': None,
    'current_feedback_time': None,
    # For LLM/TTS
    'pending_rep_analysis': None,
    'llm_feedback': {},
    'spoken_reps': set(),
    'last_llm_call_time': None,
    'latest_llm_feedback': None,
    'feedback_ready': False,
    # Timing
    'session_start_time': None,
    # MODIFIED: Track if first LLM call has been made
    'first_llm_call_made': False,
    # MODIFIED: Track which rep was last sent to LLM to avoid duplicates
    'last_rep_sent_to_llm': 0,
}


def reset_state_tracker():
    """Reset all per-run state in the global state_tracker."""
    state_tracker['state_seq'] = []
    state_tracker['prev_state'] = None
    state_tracker['curr_state'] = None
    state_tracker['SQUAT_COUNT'] = 0
    state_tracker['angles_during_rep'] = {'spine': [], 'knee': [], 'ankle': []}
    state_tracker['rep_data'] = []
    state_tracker['current_rep_issues'] = []
    state_tracker['current_rep_score'] = None
    state_tracker['show_feedback'] = False
    state_tracker['total_issues'] = 0
    state_tracker['total_compliance'] = 0
    state_tracker['feedback_start_time'] = None
    state_tracker['current_feedback_time'] = None
    state_tracker['pending_rep_analysis'] = None
    state_tracker['llm_feedback'] = {}
    state_tracker['spoken_reps'] = set()
    state_tracker['last_llm_call_time'] = None
    state_tracker['latest_llm_feedback'] = None
    state_tracker['feedback_ready'] = False
    state_tracker['session_start_time'] = None
    state_tracker['first_llm_call_made'] = False
    state_tracker['last_rep_sent_to_llm'] = 0


# ===================== GEOMETRY / ANALYSIS FUNCTIONS =====================

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def draw_dotted_line(frame, point, start, end, line_color):
    x = point[0]
    for y in range(start, end, 10):
        if y + 5 < end:
            cv2.line(frame, (x, y), (x, y + 5), line_color, 2)


def get_state(knee_angle):
    knee = None
    if THRESHOLDS['HIP_KNEE_VERT']['NORMAL'][0] <= knee_angle <= THRESHOLDS['HIP_KNEE_VERT']['NORMAL'][1]:
        knee = 1
    elif THRESHOLDS['HIP_KNEE_VERT']['TRANS'][0] <= knee_angle <= THRESHOLDS['HIP_KNEE_VERT']['TRANS'][1]:
        knee = 2
    elif THRESHOLDS['HIP_KNEE_VERT']['PASS'][0] <= knee_angle <= THRESHOLDS['HIP_KNEE_VERT']['PASS'][1]:
        knee = 3
    return f's{knee}' if knee else None


def analyze_form_and_score(max_spine, min_spine, max_knee, max_ankle):
    """
    Analyze form, return issues with deviations and calculate score.

    Uses continuous conditions for spine/knee/ankle to give a 0-1 score each.
    """
    issues = []
    deviations = {}
    abs_deviations = {
        'bend_forward': 0,
        'bend_backward': 0,
        'shallow_squat': 0,
        'deep_squat': 0,
        'knee_crossing_toe': 0
    }

    spine_condition = 1
    knee_condition = 1
    ankle_condition = 1

    # Spine
    if min_spine <= 10:
        deviation = 10 - min_spine
        issues.append("Bend forward")
        deviations['bend_forward'] = deviation
        abs_deviations['bend_forward'] = deviation
        spine_condition = (min_spine - 0) / (10 - 0)

    if max_spine >= 40:
        deviation = max_spine - 40
        issues.append("Bend backward")
        deviations['bend_backward'] = deviation
        abs_deviations['bend_backward'] = deviation

        if max_spine > 50:
            spine_condition = 0
        else:
            spine_condition = (50 - max_spine) / (50 - 40)

    # Knee depth
    if max_knee <= 75:
        deviation = 75 - max_knee
        issues.append("Shallow squat")
        deviations['shallow_squat'] = deviation
        abs_deviations['shallow_squat'] = deviation

        if max_knee < 55:
            knee_condition = 0
        else:
            knee_condition = (max_knee - 55) / (80 - 55)  # interpolation
    elif max_knee >= 90:
        deviation = max_knee - 90
        issues.append("Deep squat")
        deviations['deep_squat'] = deviation
        abs_deviations['deep_squat'] = deviation

        if max_knee > 110:
            knee_condition = 0
        else:
            knee_condition = (110 - max_knee) / (110 - 90)

    # Ankle
    if max_ankle >= 35:
        deviation = max_ankle - 35
        issues.append("Knee crossing toe")
        deviations['knee_crossing_toe'] = deviation
        abs_deviations['knee_crossing_toe'] = deviation

        if max_ankle > 40:
            ankle_condition = 0
        else:
            ankle_condition = (40 - max_ankle) / (40 - 35)

    score = (0.45 * spine_condition) + (0.35 * knee_condition) + (0.20 * ankle_condition)
    return issues, deviations, abs_deviations, score, spine_condition, knee_condition, ankle_condition


def check_compliance(current_rep_data, previous_rep_data):
    """Compare current vs previous rep deviations to see improvements."""
    compliances = []
    compliance_count = 0

    curr_abs_dev = current_rep_data['abs_deviations']
    prev_abs_dev = previous_rep_data['abs_deviations']

    error_types = ['bend_forward', 'bend_backward', 'shallow_squat', 'deep_squat', 'knee_crossing_toe']
    error_names = {
        'bend_forward': 'Bend forward',
        'bend_backward': 'Bend backward',
        'shallow_squat': 'Shallow squat',
        'deep_squat': 'Deep squat',
        'knee_crossing_toe': 'Knee crossing toe'
    }

    for error_type in error_types:
        prev_dev = prev_abs_dev[error_type]
        curr_dev = curr_abs_dev[error_type]
        if prev_dev > 0 and curr_dev < prev_dev:
            improvement = prev_dev - curr_dev
            compliances.append({
                'error': error_names[error_type],
                'prev_deviation': prev_dev,
                'curr_deviation': curr_dev,
                'improvement': improvement
            })
            compliance_count += 1

    return compliances, compliance_count


def update_state_sequence(state, spine_angle, knee_angle, ankle_angle):
    if state in ['s2', 's3']:
        state_tracker['angles_during_rep']['spine'].append(spine_angle)
        state_tracker['angles_during_rep']['knee'].append(knee_angle)
        state_tracker['angles_during_rep']['ankle'].append(ankle_angle)

    if state == 's2':
        if (('s3' not in state_tracker['state_seq']) and (state_tracker['state_seq'].count('s2') == 0)) or \
           (('s3' in state_tracker['state_seq']) and (state_tracker['state_seq'].count('s2') == 1)):
            state_tracker['state_seq'].append(state)

    elif state == 's3':
        if (state not in state_tracker['state_seq']) and 's2' in state_tracker['state_seq']:
            state_tracker['state_seq'].append(state)


# ===================== LLM (OLLAMA) FUNCTIONS =====================

def call_ollama(prompt: str) -> str:
    if not USE_OLLAMA:
        return ""
    
    # Log time since start when the HTTP call is started
    call_time = time.time()
    start = state_tracker.get('session_start_time')
    if start is not None:
        t_from_start = call_time - start
    else:
        t_from_start = 0.0
    print(f"[{t_from_start:.1f}s] 🔔 LLM HTTP call started")

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 100
        },
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()  # will raise for non-2xx
        data = resp.json()
        return data.get("response", "").strip()
    except Exception as e:
        print(f"[Ollama error] {e}")
        return ""


def generate_llm_feedback(last_rep: Dict) -> str:
    """
    Generate feedback based on the last rep's spine/knee/ankle condition scores
    and its form issues. No buffering; single-rep only.
    """
    if not last_rep:
        return ""

    spine_cond = float(last_rep['spine_cond'])
    knee_cond  = float(last_rep['knee_cond'])
    ankle_cond = float(last_rep['ankle_cond'])
    rep_id     = last_rep.get('rep', None)
    issues     = last_rep.get('form_issues', []) or []
    issues_str = ", ".join(issues) if issues else "None (form looked clean)"

    # Convert 1–10 scores to 0–1 for the prompt
    b01 = {k: v / 10.0 for k, v in BIG5_PERSONALITY.items()}

    prompt = f"""
""You are a concise squat coach. Given spine, knee, and ankle conditions (0-1), Big Five personality traits (0-1), and a list of issues, identify the priority area and give 1-2 short cues. Max 18 words."

Last rep{f" (rep #{rep_id})" if rep_id is not None else ""}:
 Spine : {spine_cond:.2f},Knee :  {knee_cond:.2f},Ankle : {ankle_cond:.2f},Openness: {b01['openness']:.1f},Conscientiousness: {b01['conscientiousness']:.1f},Extraversion: {b01['extraversion']:.1f},Agreeableness: {b01['agreeableness']:.1f},Neuroticism: {b01['neuroticism']:.1f},issues: {issues_str}
 
"""
    return call_ollama(prompt)


# ===================== TTS (pyttsx3) =====================

def _speak_text_blocking(text: str):
    if not text or not USE_TTS:
        return
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", TTS_RATE)
        engine.setProperty("volume", TTS_VOLUME)
        if TTS_VOICE_INDEX is not None:
            voices = engine.getProperty("voices")
            if 0 <= TTS_VOICE_INDEX < len(voices):
                engine.setProperty("voice", voices[TTS_VOICE_INDEX].id)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"[TTS error] {e}")


def speak_async(text: str):
    if not USE_TTS or not text:
        return
    t = threading.Thread(target=_speak_text_blocking, args=(text,), daemon=True)
    t.start()


# ===================== ASYNC LLM WORKER =====================

llm_queue: "Queue[Dict]" = Queue()


def feedback_worker():
    """Worker thread that processes last-rep summaries."""
    while True:
        last_rep = llm_queue.get()
        try:
            if not USE_OLLAMA or not last_rep:
                continue
            text = generate_llm_feedback(last_rep)
            if text:
                state_tracker['latest_llm_feedback'] = text
                state_tracker['feedback_ready'] = True
                print(f"[LLM Feedback Ready]: {text}\n")
        finally:
            llm_queue.task_done()


threading.Thread(target=feedback_worker, daemon=True).start()


# ===================== HELPER: Send rep to LLM =====================

def send_rep_to_llm(rep_data: Dict, frame_time: float, reason: str = ""):
    """Helper function to send rep data to LLM and update tracking."""
    if not USE_OLLAMA or not rep_data:
        return
    
    rep_num = rep_data.get('rep', 0)
    
    # Avoid sending the same rep twice in a row
    if rep_num == state_tracker['last_rep_sent_to_llm']:
        print(f"[{frame_time:.1f}s] Rep #{rep_num} already sent to LLM, skipping duplicate.")
        return
    
    llm_queue.put(rep_data)
    state_tracker['last_llm_call_time'] = time.time()
    state_tracker['last_rep_sent_to_llm'] = rep_num
    
    reason_str = f" ({reason})" if reason else ""
    print(f"\n📤 [{frame_time:.1f}s] Sending Rep #{rep_num} to LLM{reason_str}...")


# ===================== MAIN PROCESSING FUNCTION =====================

def run_squat_analyzer(video_path: Optional[str] = None):
    reset_state_tracker()

    if video_path is None:
        is_live = True
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open webcam (device 0).")
            input("Press Enter to exit...")
            return
        print("🎥 Squat Form Analyzer with Timed LLM Feedback (Webcam)")
        print("="*60)
        print(f"First LLM call on first rep, then every {LLM_INTERVAL_SECONDS:.0f}s with latest rep.")
        print("Press 'q' to quit.\n")
    else:
        is_live = False
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file: {video_path}")
            input("Press Enter to exit...")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        print("🎥 Squat Form Analyzer with Timed LLM Feedback (Video)")
        print("="*60)
        print(f"Video: {video_path}")
        print(f"FPS: {fps:.2f} | Frames: {frame_count} | Duration: {duration:.1f}s")
        print(f"First LLM call on first rep, then every {LLM_INTERVAL_SECONDS:.0f}s with latest rep.")
        print("="*60)
        print("Processing video...\n")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 540, 960)

    frame_number = 0
    session_start_time = time.time()
    state_tracker['session_start_time'] = session_start_time
    fps = cap.get(cv2.CAP_PROP_FPS) or 0

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if is_live:
                    print("\n❌ Camera read error or stream ended.")
                else:
                    print("\n✅ Video processing completed!")
                break

            frame_number += 1
            current_wall_time = time.time()
            
            # Time since the start of the run or video
            if is_live:
                frame_time = current_wall_time - session_start_time
            else:
                frame_time = frame_number / fps if fps > 0 else 0

            # MODIFIED: Timed LLM calls after the first rep
            # After first call is made, check every frame if 12s has passed
            if USE_OLLAMA and state_tracker['first_llm_call_made']:
                elapsed_since_llm = current_wall_time - state_tracker['last_llm_call_time']
                if elapsed_since_llm >= LLM_INTERVAL_SECONDS:
                    # Send the latest rep data
                    if state_tracker['rep_data']:
                        latest_rep = state_tracker['rep_data'][-1]
                        send_rep_to_llm(latest_rep, frame_time, reason="12s interval")

            frame_height, frame_width, _ = frame.shape

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                left_shoulder = [
                    int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame_width),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame_height)
                ]
                left_hip = [
                    int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * frame_width),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * frame_height)
                ]
                left_knee = [
                    int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * frame_width),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * frame_height)
                ]
                left_ankle = [
                    int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * frame_width),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * frame_height)
                ]

                vertical_point_hip = np.array([left_hip[0], 0])
                vertical_point_knee = np.array([left_knee[0], 0])
                vertical_point_ankle = np.array([left_ankle[0], 0])

                spine_angle = calculate_angle(vertical_point_hip, left_hip, left_shoulder)
                knee_angle = calculate_angle(vertical_point_knee, left_knee, left_hip)
                ankle_angle = calculate_angle(vertical_point_ankle, left_ankle, left_knee)

                current_state = get_state(int(knee_angle))
                prev_state = state_tracker['prev_state']
                state_tracker['curr_state'] = current_state

                # Feedback viewing time: leaving s1 -> s2 (start of next rep)
                if prev_state == 's1' and current_state == 's2':
                    if state_tracker['feedback_start_time'] is not None:
                        feedback_duration = frame_time - state_tracker['feedback_start_time']
                        state_tracker['current_feedback_time'] = feedback_duration
                        state_tracker['feedback_start_time'] = None

                # Feedback visibility & speaking LLM feedback
                if current_state == 's1':
                    state_tracker['show_feedback'] = True
                    # Speak feedback if ready and not already spoken
                    if (state_tracker['feedback_ready'] and 
                        state_tracker['latest_llm_feedback'] and
                        state_tracker['latest_llm_feedback'] not in state_tracker['spoken_reps']):
                        speak_async(state_tracker['latest_llm_feedback'])
                        state_tracker['spoken_reps'].add(state_tracker['latest_llm_feedback'])
                        state_tracker['feedback_ready'] = False
                elif current_state in ['s2', 's3']:
                    state_tracker['show_feedback'] = False

                # Update angles & state sequence
                update_state_sequence(current_state, spine_angle, knee_angle, ankle_angle)

                # s3 -> s2: freeze metrics (pre-compute for performance)
                if prev_state == 's3' and current_state == 's2':
                    angles_rep = state_tracker['angles_during_rep']
                    if len(angles_rep['spine']) > 0:
                        max_spine = max(angles_rep['spine'])
                        max_knee = max(angles_rep['knee'])
                        max_ankle = max(angles_rep['ankle'])
                        min_spine = min(angles_rep['spine'])
                        min_knee = min(angles_rep['knee'])
                        min_ankle = min(angles_rep['ankle'])

                        form_issues, deviations, abs_deviations, score, spine_cond, knee_cond, ankle_cond = (
                            analyze_form_and_score(max_spine, min_spine, max_knee, max_ankle)
                        )
                        issue_count = len(form_issues)

                        rep_index = state_tracker['SQUAT_COUNT'] + 1
                        rep_summary = {
                            'rep': rep_index,
                            'max_spine': max_spine,
                            'max_knee': max_knee,
                            'max_ankle': max_ankle,
                            'min_spine': min_spine,
                            'min_knee': min_knee,
                            'min_ankle': min_ankle,
                            'form_issues': form_issues,
                            'deviations': deviations,
                            'abs_deviations': abs_deviations,
                            'issue_count': issue_count,
                            'score': score,
                            'spine_cond': spine_cond,
                            'knee_cond': knee_cond,
                            'ankle_cond': ankle_cond,
                        }
                        state_tracker['pending_rep_analysis'] = rep_summary

                # s1 + full sequence: finalize rep
                if current_state == 's1':
                    if len(state_tracker['state_seq']) == 3:
                        rep_index = state_tracker['SQUAT_COUNT'] + 1
                        rep_info = state_tracker['pending_rep_analysis']

                        # If pre-computed rep doesn't match, compute now
                        if rep_info is None or rep_info.get('rep') != rep_index:
                            angles_rep = state_tracker['angles_during_rep']
                            if len(angles_rep['spine']) > 0:
                                max_spine = max(angles_rep['spine'])
                                max_knee = max(angles_rep['knee'])
                                max_ankle = max(angles_rep['ankle'])
                                min_spine = min(angles_rep['spine'])
                                min_knee = min(angles_rep['knee'])
                                min_ankle = min(angles_rep['ankle'])
                                form_issues, deviations, abs_deviations, score, spine_cond, knee_cond, ankle_cond = (
                                    analyze_form_and_score(max_spine, min_spine, max_knee, max_ankle)
                                )
                                issue_count = len(form_issues)
                                
                                rep_info = {
                                    'rep': rep_index,
                                    'max_spine': max_spine,
                                    'max_knee': max_knee,
                                    'max_ankle': max_ankle,
                                    'min_spine': min_spine,
                                    'min_knee': min_knee,
                                    'min_ankle': min_ankle,
                                    'form_issues': form_issues,
                                    'deviations': deviations,
                                    'abs_deviations': abs_deviations,
                                    'issue_count': issue_count,
                                    'score': score,
                                    'spine_cond': spine_cond,
                                    'knee_cond': knee_cond,
                                    'ankle_cond': ankle_cond,
                                }

                        if rep_info:
                            state_tracker['SQUAT_COUNT'] = rep_index
                            
                            # Compliance & feedback viewing time
                            compliances = []
                            compliance_count = 0
                            feedback_time = None
                            
                            if state_tracker['SQUAT_COUNT'] > 1:
                                previous_rep = state_tracker['rep_data'][-1]
                                current_rep_temp = {'abs_deviations': rep_info['abs_deviations']}
                                compliances, compliance_count = check_compliance(current_rep_temp, previous_rep)
                                state_tracker['total_compliance'] += compliance_count

                                if state_tracker['current_feedback_time'] is not None:
                                    feedback_time = state_tracker['current_feedback_time']
                                    state_tracker['current_feedback_time'] = None

                            state_tracker['current_rep_issues'] = rep_info['form_issues']
                            state_tracker['current_rep_score'] = rep_info['score']

                            # Start timing feedback display for this rep
                            if state_tracker['SQUAT_COUNT'] > 0:
                                state_tracker['feedback_start_time'] = frame_time

                            # Store rep data (must be done before LLM call so rep_data is available)
                            full_rep_data = {
                                **rep_info,
                                'compliances': compliances,
                                'compliance_count': compliance_count,
                                'feedback_time': feedback_time,
                                'time': frame_time
                            }
                            state_tracker['rep_data'].append(full_rep_data)

                            state_tracker['total_issues'] += rep_info['issue_count']

                            # MODIFIED: First rep triggers first LLM call immediately
                            if USE_OLLAMA and not state_tracker['first_llm_call_made']:
                                send_rep_to_llm(full_rep_data, frame_time, reason="first rep")
                                state_tracker['first_llm_call_made'] = True

                            # Detailed per-rep printout
                            print(f"Rep #{state_tracker['SQUAT_COUNT']} @ {frame_time:.1f}s:")
                            print(f"  Score: {rep_info['score']:.2f}")

                            if state_tracker['SQUAT_COUNT'] > 1 and feedback_time is not None:
                                print(f"  Feedback viewing time: {feedback_time:.2f}s")

                            if rep_info['form_issues']:
                                print(f"  ⚠️  Form Issues ({rep_info['issue_count']}):")
                                dev = rep_info['deviations']
                                for issue in rep_info['form_issues']:
                                    line = f"      - {issue}"
                                    if 'Bend forward' in issue and 'bend_forward' in dev:
                                        line += f" (Need {dev['bend_forward']:.1f}° more)"
                                    elif 'Bend backward' in issue and 'bend_backward' in dev:
                                        line += f" (Reduce {dev['bend_backward']:.1f}°)"
                                    elif 'Shallow squat' in issue and 'shallow_squat' in dev:
                                        line += f" (Go {dev['shallow_squat']:.1f}° deeper)"
                                    elif 'Deep squat' in issue and 'deep_squat' in dev:
                                        line += f" (Reduce {dev['deep_squat']:.1f}°)"
                                    elif 'Knee crossing toe' in issue and 'knee_crossing_toe' in dev:
                                        line += f" (Reduce {dev['knee_crossing_toe']:.1f}°)"
                                    print(line)
                            else:
                                print("  ✅ Perfect form!")

                            if state_tracker['SQUAT_COUNT'] > 1:
                                if compliances:
                                    print(f"  ✅ Compliance ({compliance_count}):")
                                    for comp in compliances:
                                        print(f"      - {comp['error']}: Improved by {comp['improvement']:.1f}° "
                                              f"({comp['prev_deviation']:.1f}° → {comp['curr_deviation']:.1f}°)")
                                else:
                                    print("  ❌ No compliance with previous feedback")

                            print()

                            state_tracker['pending_rep_analysis'] = None

                    # reset sequence and angles
                    state_tracker['state_seq'] = []
                    state_tracker['angles_during_rep'] = {'spine': [], 'knee': [], 'ankle': []}

                state_tracker['prev_state'] = current_state

                # ===================== Visualization =====================

                draw_dotted_line(image, left_hip, max(0, left_hip[1] - 100),
                                 min(frame_height, left_hip[1] + 50), COLORS['blue'])
                draw_dotted_line(image, left_knee, max(0, left_knee[1] - 100),
                                 min(frame_height, left_knee[1] + 50), COLORS['blue'])
                draw_dotted_line(image, left_ankle, max(0, left_ankle[1] - 100),
                                 min(frame_height, left_ankle[1] + 50), COLORS['blue'])

                spine_multiplier = 1 if left_shoulder[0] > left_hip[0] else -1
                knee_multiplier = 1 if left_hip[0] > left_knee[0] else -1
                ankle_multiplier = 1 if left_knee[0] > left_ankle[0] else -1

                cv2.ellipse(image, tuple(left_hip), (40, 40),
                            angle=0, startAngle=-90,
                            endAngle=-90 + spine_multiplier * int(spine_angle),
                            color=COLORS['white'], thickness=3)
                cv2.ellipse(image, tuple(left_knee), (40, 40),
                            angle=0, startAngle=-90,
                            endAngle=-90 + knee_multiplier * int(knee_angle),
                            color=COLORS['white'], thickness=3)
                cv2.ellipse(image, tuple(left_ankle), (40, 40),
                            angle=0, startAngle=-90,
                            endAngle=-90 + ankle_multiplier * int(ankle_angle),
                            color=COLORS['white'], thickness=3)

                cv2.line(image, tuple(left_hip), tuple(left_shoulder), COLORS['light_blue'], 4)
                cv2.line(image, tuple(left_hip), tuple(left_knee), COLORS['light_blue'], 4)
                cv2.line(image, tuple(left_knee), tuple(left_ankle), COLORS['light_blue'], 4)

                cv2.circle(image, tuple(left_shoulder), 7, COLORS['yellow'], -1)
                cv2.circle(image, tuple(left_hip), 7, COLORS['yellow'], -1)
                cv2.circle(image, tuple(left_knee), 7, COLORS['yellow'], -1)
                cv2.circle(image, tuple(left_ankle), 7, COLORS['yellow'], -1)

                cv2.putText(image, f'{int(spine_angle)}',
                            (left_hip[0] + 15, left_hip[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS['green'], 3)
                cv2.putText(image, f'{int(knee_angle)}',
                            (left_knee[0] + 15, left_knee[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS['green'], 3)
                cv2.putText(image, f'{int(ankle_angle)}',
                            (left_ankle[0] + 15, left_ankle[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS['green'], 3)

                cv2.putText(image, f'SPINE: {int(spine_angle)}',
                            (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['white'], 2)
                cv2.putText(image, f'KNEE: {int(knee_angle)}',
                            (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['white'], 2)
                cv2.putText(image, f'ANKLE: {int(ankle_angle)}',
                            (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['white'], 2)

                cv2.putText(image, f"REPS: {state_tracker['SQUAT_COUNT']}",
                            (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLORS['cyan'], 4)

                # MODIFIED: Show LLM status
                if USE_OLLAMA:
                    if not state_tracker['first_llm_call_made']:
                        # Waiting for first rep
                        status_text = "LLM: Waiting for first rep..."
                        status_color = COLORS['yellow']
                    else:
                        elapsed_since_llm = current_wall_time - state_tracker['last_llm_call_time']
                        time_until_next = max(0, LLM_INTERVAL_SECONDS - elapsed_since_llm)
                        
                        if time_until_next > 0:
                            status_text = f"Next LLM in: {time_until_next:.1f}s"
                            status_color = COLORS['orange']
                        else:
                            status_text = "LLM Ready (sending on next frame)"
                            status_color = COLORS['green']
                    
                    cv2.putText(image, status_text,
                                (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

                state_name_map = {'s1': 'STANDING', 's2': 'TRANSITION', 's3': 'SQUAT'}
                state_display = state_name_map.get(current_state, 'UNKNOWN')
                state_color = COLORS['green'] if current_state == 's3' else COLORS['white']
                cv2.putText(image, f"STATE: {state_display}",
                            (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)

                # Show feedback (current rep issues + LLM feedback)
                if state_tracker['show_feedback'] and state_tracker['SQUAT_COUNT'] > 0:
                    right_x = frame_width - 350
                    
                    # Current rep issues
                    cv2.putText(image, "LAST REP ISSUES:",
                                (right_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['orange'], 2)

                    if state_tracker['current_rep_issues']:
                        y_offset = 80
                        for issue in state_tracker['current_rep_issues']:
                            hint = ISSUE_HINTS.get(issue,issue)
                            display_text = f"- {hint}"
                            
                            cv2.putText(image, display_text,
                                        (right_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['red'], 2)
                            y_offset += 35
                    else:
                        cv2.putText(image, "Good form!",
                                    (right_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['green'], 2)

                    # Score (still shown locally; not sent to LLM)
                    if state_tracker['current_rep_score'] is not None:
                        score = state_tracker['current_rep_score']
                        if score >= 0.8:
                            score_color = COLORS['green']
                        elif score >= 0.5:
                            score_color = COLORS['yellow']
                        else:
                            score_color = COLORS['red']
                        cv2.putText(image, f"SCORE: {score:.2f}",
                                    (right_x, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.2, score_color, 3)
                    
                    # LLM Feedback
                    '''if state_tracker['latest_llm_feedback']:
                        cv2.putText(image, "COACH ADVICE:",
                                    (right_x, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['cyan'], 2)
                        
                        # Word wrap the feedback
                        feedback_lines = []
                        words = state_tracker['latest_llm_feedback'].split()
                        current_line = ""
                        for word in words:
                            if len(current_line + word) < 30:
                                current_line += word + " "
                            else:
                                feedback_lines.append(current_line.strip())
                                current_line = word + " "
                        if current_line:
                            feedback_lines.append(current_line.strip())
                        
                        y_offset = 310
                        for line in feedback_lines[:3]:  # Max 3 lines
                            cv2.putText(image, line,
                                        (right_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['white'], 2)
                            y_offset += 30'''

            except Exception:
                cv2.putText(image, "No pose detected",
                            (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS['red'], 2)

            cv2.imshow(window_name, image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("\n⚠️  Stopped by user")
                break

    cap.release()
    cv2.destroyAllWindows()

    # ===================== FINAL SUMMARY =====================

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Total Reps: {state_tracker['SQUAT_COUNT']}")

    if len(state_tracker['rep_data']) > 0:
        last_rep_issues = state_tracker['rep_data'][-1]['issue_count']
        adjusted_total_issues = state_tracker['total_issues'] - last_rep_issues
    else:
        adjusted_total_issues = 0

    print(f"Total Issues (excluding last rep): {adjusted_total_issues}")
    print(f"Total Compliance: {state_tracker['total_compliance']}")

    # Average feedback viewing time (exclude None)
    feedback_times = [
        r['feedback_time']
        for r in state_tracker['rep_data']
        if r.get('feedback_time') is not None
    ]
    if feedback_times:
        avg_feedback_time = np.mean(feedback_times)
        print(f"Average Feedback Viewing Time: {avg_feedback_time:.2f}s\n")
    else:
        print("Average Feedback Viewing Time: N/A\n")

    # Detailed per-rep analysis
    if state_tracker['rep_data']:
        print("Detailed Rep Analysis:")
        print("-"*60)
        for i, rep in enumerate(state_tracker['rep_data']):
            is_last_rep = (i == len(state_tracker['rep_data']) - 1)

            print(f"Rep {rep['rep']} @ {rep['time']:.1f}s:")
            print(f"  Score: {rep['score']:.2f}")

            # Feedback viewing time per rep (from rep 2 onwards)
            if rep['rep'] > 1 and rep.get('feedback_time') is not None:
                print(f"  Feedback viewing time: {rep['feedback_time']:.2f}s")

            # Show issues & deviations (skip counting for last rep)
            if not is_last_rep:
                print(f"  Issues: {rep['issue_count']}")
                if rep['form_issues']:
                    dev = rep['deviations']
                    for issue in rep['form_issues']:
                        line = f"    - {issue}"
                        if 'Bend forward' in issue and 'bend_forward' in dev:
                            line += f" → Need {dev['bend_forward']:.1f}° more"
                        elif 'Bend backward' in issue and 'bend_backward' in dev:
                            line += f" → Reduce {dev['bend_backward']:.1f}°"
                        elif 'Shallow squat' in issue and 'shallow_squat' in dev:
                            line += f" → Go {dev['shallow_squat']:.1f}° deeper"
                        elif 'Deep squat' in issue and 'deep_squat' in dev:
                            line += f" → Reduce {dev['deep_squat']:.1f}°"
                        elif 'Knee crossing toe' in issue and 'knee_crossing_toe' in dev:
                            line += f" → Reduce {dev['knee_crossing_toe']:.1f}°"
                        print(line)
                else:
                    print("    ✅ Perfect form!")
            else:
                print("  (Last rep - issues not counted)")

            # Compliance per rep (from rep 2 onwards)
            if rep['rep'] > 1:
                print(f"  Compliance: {rep['compliance_count']}")
                if rep['compliances']:
                    for comp in rep['compliances']:
                        print(
                            f"    ✅ {comp['error']}: {comp['prev_deviation']:.1f}° → "
                            f"{comp['curr_deviation']:.1f}° (improved {comp['improvement']:.1f}°)"
                        )
                else:
                    print("    ❌ No compliance")
            print()

        avg_score = np.mean([r['score'] for r in state_tracker['rep_data']])
        print(f"Average Score: {avg_score:.2f}")
        print(f"📊 TOTAL ISSUES (excluding last rep): {adjusted_total_issues}")
        print(f"✅ TOTAL COMPLIANCE: {state_tracker['total_compliance']}")
        if feedback_times:
            print(f"⏱️  AVERAGE FEEDBACK VIEWING TIME: {np.mean(feedback_times):.2f}s")

    print("="*60)


# ===================== ENTRY POINT =====================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        # Ask for Big Five before running, if LLM is enabled
        get_big5_from_user()
        if arg.lower() == "webcam":
            run_squat_analyzer(video_path=None)
        else:
            run_squat_analyzer(video_path=arg)
    else:
        print("Select input source:")
        print("  1) Webcam")
        print("  2) Video file")
        choice = input("Enter 1 or 2: ").strip()
        if choice == "2":
            path = input("Enter video file path: ").strip()
            get_big5_from_user()
            run_squat_analyzer(video_path=path)
        else:
            get_big5_from_user()
            run_squat_analyzer(video_path=None)