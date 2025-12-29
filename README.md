# 연료펌프 모듈 검사 시스템
# Fuel Pump Module Inspection System

라즈베리파이 AI 키트(Hailo-8L)를 활용한 연료펌프 모듈 자동 품질검사 시스템

## 주요 기능

- **외관 결함 검사**: 스크래치, 찍힘, 균열, 이물질, 사출불량 검출
- **바코드 검사**: 부착 유무, 위치, 데이터 유효성 확인
- **이종품 검사**: 잘못된 부품 조립, 형상/색상 불일치 검출
- **딥러닝 검사**: YOLOv8 기반 결함 검출 (ONNX, Hailo 지원)

## 시스템 구성

```
[카메라] → [라즈베리파이 5 + Hailo-8L] → [검사 모듈]
                                              ↓
                                    [외관/바코드/이종품 검사]
                                              ↓
                                        [OK/NG 판정]
                                              ↓
                                    [PLC 신호 + DB 저장]
```

## 하드웨어 요구사항

| 구성품 | 사양 |
|--------|------|
| Raspberry Pi 5 | 8GB RAM 권장 |
| Hailo-8L AI 가속기 | 13 TOPS |
| 카메라 | 1920x1080 이상 |
| 조명 | 링라이트/백라이트 |

## 설치

```bash
# 저장소 클론
git clone https://github.com/YOUR_USERNAME/fuel_pump_inspection.git
cd fuel_pump_inspection

# 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 라즈베리파이에서 Hailo 드라이버 설치
sudo apt install hailo-all
```

## 사용법

### 기준 이미지 등록
```bash
python main.py --mode register --part-id "FP-2024-A01"
```

### 검사 실행
```bash
# 수동 트리거 (스페이스바)
python main.py --mode run --trigger manual

# GPIO 트리거 (PLC 연동)
python main.py --mode run --trigger gpio

# 자동 연속 검사
python main.py --mode run --trigger auto
```

### 테스트
```bash
python test_inspection.py
```

## 프로젝트 구조

```
fuel_pump_inspection/
├── main.py                 # 메인 실행 파일
├── config.yaml             # 설정 파일
├── requirements.txt        # 의존성 목록
├── inspectors/
│   ├── defect_inspector.py    # 외관 결함 검사
│   ├── barcode_inspector.py   # 바코드 검사
│   ├── variant_inspector.py   # 이종품 검사
│   └── deep_learning_inspector.py  # 딥러닝 검사
├── utils/
│   ├── camera.py
│   ├── gpio_control.py
│   ├── database.py
│   └── hailo_utils.py
├── scripts/
│   └── train_yolov8.py    # 모델 학습 스크립트
├── models/                 # AI 모델 파일
├── reference/              # 기준 이미지
└── logs/                   # 검사 로그
```

## 설정 (config.yaml)

```yaml
inspection:
  defect:
    enabled: true
    min_defect_area: 50
  barcode:
    enabled: true
    required_prefix: "FP"
  variant:
    enabled: true
    similarity_threshold: 0.85

deep_learning:
  enabled: true
  engine: "onnx"  # hailo, onnx, pytorch
  detection_model: "models/fuel_pump_defect.onnx"
```

## 딥러닝 모델 학습

```bash
# 데이터셋 구조 생성
python scripts/train_yolov8.py setup --dir datasets/fuel_pump_defects

# YOLOv8 학습
python scripts/train_yolov8.py train --data datasets/fuel_pump_defects/dataset.yaml --epochs 100

# ONNX 내보내기
python scripts/train_yolov8.py export --model runs/train/best.pt --format onnx
```

## 라이선스

MIT License

## 문의

프로젝트 관련 문의사항은 Issues를 통해 남겨주세요.
