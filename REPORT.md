# 운전자 상태 모니터링 및 졸음경고 시스템: 파이썬 코드 기반 기술 보고서
작성일: 2025-11-02

## 1. 서론
자율주행 기술이 보편화되더라도, 돌발 상황 대응을 위한 운전자의 개입은 여전히 중요합니다. 본 프로젝트는 웹캠 영상을 기반으로 **졸음**, **전방주시 이탈**, **스마트폰 사용 의심** 등 3가지 위험 행태를 실시간 감지하여 경고하는 경량 프로토타입을 구현합니다. (과목 제출용 데모 및 향후 고도화 기반)

### 1.1 주제 선정 배경 및 필요성
- 운전자 부주의는 교통사고의 주요 원인입니다.
- 실시간, 경량화된 판단 모듈은 임베디드/차량 내 환경에 적합합니다.

### 1.2 관련 연구/서비스 간단 비교
- EAR(Eye Aspect Ratio) 기반 졸음 감지: 계산량이 적고 실시간성 우수
- PnP 기반 헤드포즈 추정: 표준적인 카메라 포즈 추정 기법
- 대규모 객체 검출 모델(예: YOLO) 대비, 본 과제는 **MediaPipe**를 활용한 경량/무학습 추정 + 휴리스틱으로 접근

## 2. 이론적 배경 및 기술 스택
- **기술 스택:** Python, OpenCV, MediaPipe, NumPy
- **핵심 개념:**
  - EAR: 눈 둘레 6점의 세로/가로 거리비로 눈 감김 판단
  - Head Pose: 얼굴의 2D-3D 대응점을 이용한 solvePnP로 yaw/pitch/roll 추정
  - Hand Proximity: 손-얼굴 중심 거리와 손 형태(손목-검지끝 거리)의 조합 휴리스틱

## 3. 시스템 설계
### 3.1 전체 구조
**입력(웹캠)** → Face Mesh/Hands 추출 → (EAR, Pose, Hand-얼굴근접) 특징 계산 → 임계치 기반 상태머신 → **경고(비프음/오버레이)** → 이벤트 기록(CSV)

### 3.2 모듈
- `src/eye_aspect_ratio.py`: MediaPipe Face Mesh 좌표로 EAR 계산
- `src/head_pose.py`: 2D 랜드마크와 정적 3D 얼굴모델로 pose(yaw/pitch) 추정
- `src/phone_use.py`: 손 랜드마크가 얼굴 중심에 충분히 근접하고 손이 상대적으로 '집게' 형태일 때 **스마트폰 사용 의심** 판단
- `main.py`: 통합 파이프라인 및 알림/로깅

### 3.3 경량화 전략
- 대규모 CNN 검출기 대신 **랜덤 접근/휴리스틱**으로 실시간성 확보
- **프레임 기반 누적(연속 N프레임 조건)**로 오탐 억제

## 4. 구현 내용
### 4.1 데이터 및 전처리
- 외부 학습 데이터 없이, MediaPipe가 제공하는 얼굴/손 추정기로부터 2D 좌표를 사용
- 좌표 정규화(해당 프레임의 머리 크기)로 사용자/화면 거리 변화 대응

### 4.2 핵심 코드 스니펫
- EAR 계산: `compute_ears(landmarks)`
- 헤드포즈: `estimate_head_pose(landmarks, frame.shape)`
- 폰 사용 의심: `hand_near_face(nose_xy, hand_xy, head_size, threshold)`

### 4.3 실행 방법
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
- 파라미터는 `config.yaml`에서 조정

## 5. 실험 및 결과 (샘플)
- 장비: 노트북 내장카메라(720p), CPU 실행
- 평균 FPS: 15~30 (환경 의존)
- 임계치 예시: `ear_sleep_threshold=0.22`, `yaw_abs_threshold=25°`, `hand_face_distance_norm=0.18`
- 관찰: 눈을 길게 감거나 고개를 좌우로 크게 돌리면 경고 문구와 비프음 발생
- 한계: 실제 운전 상황(진동, 조명 변화, 안경 반사, 마스크 등)에서 오탐/미탐 가능

## 6. 결론 및 향후 개선
- **결론:** 학습 데이터 없이도 실시간으로 핵심 위험행동의 신호를 감지하는 경량 프로토타입을 구현했습니다.
- **향후 개선 방향:**
  1) 휴리스틱 폰 감지 → **소형 객체검출(TFLite/YOLO-Nano)**로 대체  
  2) 조명 변화 대응을 위한 **적응형 임계값/노출 보정**  
  3) **멀티센서(적외선, 심박 등)** 융합으로 강인성 향상  
  4) **임베디드 보드(Nano/Xavier)** 포팅 및 최적화

## 부록 A. 파일 구성
```
driver_monitoring/
├─ main.py
├─ config.yaml
├─ requirements.txt
├─ README.md
└─ src/
   ├─ eye_aspect_ratio.py
   ├─ head_pose.py
   ├─ phone_use.py
   └─ utils.py
```
