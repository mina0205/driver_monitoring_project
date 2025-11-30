# Driver Monitoring System (Python + Dlib + OpenCV)

본 프로젝트는 웹캠 기반으로 **운전자 상태(졸음, 시선 이탈)를 실시간으로 감지**하는 경량 운전자 모니터링 시스템입니다.  
Dlib 기반 얼굴 랜드마크(68점), EAR(Eye Aspect Ratio), Head Pose Estimation을 활용하여 위험 상황을 파악하고 경고음을 발생시킵니다.

---

## ✨ 주요 기능 (Features)

### ✔️ 1. 졸음 감지 (Drowsiness Detection)
- Dlib 68 landmark 중 눈 주변 6점을 이용하여 EAR 계산
- EAR이 일정 임계값 아래로 일정 시간 유지되면 `DROWSINESS DETECTED` 경고 출력
- sounddevice 기반 비프음(Beep) 출력

### ✔️ 2. 시선 이탈 감지 (Head Pose Estimation)
- solvePnP + 3D 얼굴 모델을 활용해 `Yaw`, `Pitch`, `Roll` 계산
- 사전 보정된 오프셋(Yaw/Pitch)을 기준으로  
  시선이 전방에서 벗어났을 경우 `LOOKING AWAY` 경고

### ✔️ 3. 실시간 랜드마크 시각화
- 눈, 코, 얼굴 특징점(68개) 시각화
- EAR 값, Yaw/Pitch 값, FPS, 상태 메시지가 화면에 표시됨

### ✔️ 4. 경고 이벤트 로그 저장
- `events_log.csv` 파일로 시간/이벤트 저장

---

## 🧠 핵심 알고리즘 설명

### 🔹 EAR(Eye Aspect Ratio)

- EAR 감소 → 눈 감김  
- EAR 일정 임계값 이하 → 졸음 의심

---

### 🔹 SolvePnP 기반 Head Pose
아래 6개의 Dlib 랜드마크를 이용해 solvePnP로 회전벡터(Rvec), 이동벡터(Tvec) 계산:

| 부위 | Dlib Index |
|------|------------|
| Nose Tip | 30 |
| Chin | 8 |
| Left Eye Corner | 36 |
| Right Eye Corner | 45 |
| Left Mouth Corner | 48 |
| Right Mouth Corner | 54 |

Yaw/Pitch/Roll → 오일러각 변환  
Yaw·Pitch가 기준값을 벗어나면 시선 이탈 감지.

---

## ⚙️ 실행 방법 (Run)

### 1) 가상환경 활성화
```bash
conda activate driver_env
/opt/anaconda3/envs/driver_env/bin/python main.py




