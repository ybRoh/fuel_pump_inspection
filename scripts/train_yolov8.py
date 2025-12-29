#!/usr/bin/env python3
"""
YOLOv8 결함 검출 모델 학습 스크립트
Train YOLOv8 Defect Detection Model

사용법:
    python scripts/train_yolov8.py --data datasets/defects.yaml --epochs 100

필요 패키지:
    pip install ultralytics
"""

import os
import argparse
import yaml
from pathlib import Path


def create_dataset_yaml(dataset_dir: str, classes: list, output_path: str) -> str:
    """
    데이터셋 YAML 파일 생성

    Args:
        dataset_dir: 데이터셋 디렉토리
        classes: 클래스 목록
        output_path: 출력 파일 경로

    Returns:
        생성된 YAML 파일 경로
    """
    dataset_config = {
        'path': os.path.abspath(dataset_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {i: name for i, name in enumerate(classes)}
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)

    print(f"데이터셋 설정 파일 생성: {output_path}")
    return output_path


def train_yolov8(data_yaml: str,
                 model_size: str = 's',
                 epochs: int = 100,
                 imgsz: int = 640,
                 batch: int = 16,
                 device: str = '0',
                 project: str = 'runs/train',
                 name: str = 'fuel_pump_defect') -> str:
    """
    YOLOv8 모델 학습

    Args:
        data_yaml: 데이터셋 YAML 경로
        model_size: 모델 크기 (n, s, m, l, x)
        epochs: 학습 에폭 수
        imgsz: 입력 이미지 크기
        batch: 배치 크기
        device: 학습 장치 (0: GPU, cpu: CPU)
        project: 프로젝트 디렉토리
        name: 실험 이름

    Returns:
        학습된 모델 경로
    """
    try:
        from ultralytics import YOLO

        print("\n" + "=" * 50)
        print("YOLOv8 결함 검출 모델 학습")
        print("=" * 50)

        # 모델 로드 (사전학습 가중치)
        model_name = f'yolov8{model_size}.pt'
        print(f"베이스 모델: {model_name}")

        model = YOLO(model_name)

        # 학습 시작
        print(f"\n학습 설정:")
        print(f"  데이터셋: {data_yaml}")
        print(f"  에폭: {epochs}")
        print(f"  이미지 크기: {imgsz}")
        print(f"  배치 크기: {batch}")
        print(f"  장치: {device}")

        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            patience=20,           # Early stopping
            save=True,             # 체크포인트 저장
            save_period=10,        # 10 에폭마다 저장
            cache=True,            # 데이터 캐싱
            workers=4,             # 데이터 로더 워커
            pretrained=True,       # 사전학습 가중치 사용
            optimizer='auto',      # 옵티마이저 자동 선택
            verbose=True,
            seed=42,
            deterministic=True,
            amp=True,              # Mixed precision
            # 데이터 증강
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=2.0,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
        )

        # 최종 모델 경로
        best_model = os.path.join(project, name, 'weights', 'best.pt')
        print(f"\n학습 완료!")
        print(f"최적 모델: {best_model}")

        return best_model

    except ImportError:
        print("Error: ultralytics 패키지가 설치되지 않았습니다.")
        print("설치: pip install ultralytics")
        return None


def export_model(model_path: str, format: str = 'onnx',
                 imgsz: int = 640) -> str:
    """
    모델 내보내기

    Args:
        model_path: 학습된 모델 경로
        format: 내보내기 형식 (onnx, openvino, tflite 등)
        imgsz: 입력 이미지 크기

    Returns:
        내보낸 모델 경로
    """
    try:
        from ultralytics import YOLO

        model = YOLO(model_path)

        print(f"\n모델 내보내기: {format}")
        exported = model.export(format=format, imgsz=imgsz, simplify=True)

        print(f"내보내기 완료: {exported}")
        return exported

    except Exception as e:
        print(f"내보내기 실패: {e}")
        return None


def validate_model(model_path: str, data_yaml: str) -> dict:
    """
    모델 검증

    Args:
        model_path: 모델 경로
        data_yaml: 데이터셋 YAML 경로

    Returns:
        검증 결과
    """
    try:
        from ultralytics import YOLO

        model = YOLO(model_path)
        results = model.val(data=data_yaml)

        print("\n=== 검증 결과 ===")
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")

        return {
            'map50': results.box.map50,
            'map': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr
        }

    except Exception as e:
        print(f"검증 실패: {e}")
        return {}


def create_sample_dataset_structure(base_dir: str, classes: list) -> None:
    """
    샘플 데이터셋 디렉토리 구조 생성

    Args:
        base_dir: 기본 디렉토리
        classes: 클래스 목록
    """
    dirs = [
        'images/train', 'images/val', 'images/test',
        'labels/train', 'labels/val', 'labels/test'
    ]

    for d in dirs:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)

    # 클래스 정보 파일
    classes_file = os.path.join(base_dir, 'classes.txt')
    with open(classes_file, 'w') as f:
        for cls in classes:
            f.write(f"{cls}\n")

    # README
    readme = f"""# 연료펌프 결함 검출 데이터셋

## 디렉토리 구조
```
{base_dir}/
├── images/
│   ├── train/    # 학습 이미지 (*.jpg, *.png)
│   ├── val/      # 검증 이미지
│   └── test/     # 테스트 이미지
├── labels/
│   ├── train/    # 학습 라벨 (*.txt)
│   ├── val/      # 검증 라벨
│   └── test/     # 테스트 라벨
└── classes.txt   # 클래스 목록
```

## 클래스 목록
{chr(10).join(f'{i}. {cls}' for i, cls in enumerate(classes))}

## 라벨 형식 (YOLO format)
각 이미지에 대해 동일한 이름의 .txt 파일 생성:
```
<class_id> <x_center> <y_center> <width> <height>
```
- 모든 값은 0~1 사이로 정규화
- 예: `0 0.5 0.5 0.2 0.1` (클래스 0, 중심 50%, 크기 20%x10%)

## 데이터 수집 권장량
- 클래스당 최소 100장
- 권장 500장 이상
- 다양한 조명, 각도, 배경 포함

## 라벨링 도구
- Roboflow (https://roboflow.com) - 웹 기반, 무료
- LabelImg - 로컬 설치
- CVAT - 웹 기반, 오픈소스
"""
    readme_file = os.path.join(base_dir, 'README.md')
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme)

    print(f"데이터셋 디렉토리 구조 생성: {base_dir}")
    print(f"클래스 파일: {classes_file}")
    print(f"가이드: {readme_file}")


def main():
    parser = argparse.ArgumentParser(
        description='YOLOv8 결함 검출 모델 학습',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='명령어')

    # 학습 명령
    train_parser = subparsers.add_parser('train', help='모델 학습')
    train_parser.add_argument('--data', type=str, required=True, help='데이터셋 YAML')
    train_parser.add_argument('--model', type=str, default='s', help='모델 크기 (n/s/m/l/x)')
    train_parser.add_argument('--epochs', type=int, default=100, help='에폭 수')
    train_parser.add_argument('--imgsz', type=int, default=640, help='이미지 크기')
    train_parser.add_argument('--batch', type=int, default=16, help='배치 크기')
    train_parser.add_argument('--device', type=str, default='0', help='장치')
    train_parser.add_argument('--name', type=str, default='fuel_pump_defect', help='실험 이름')

    # 내보내기 명령
    export_parser = subparsers.add_parser('export', help='모델 내보내기')
    export_parser.add_argument('--model', type=str, required=True, help='모델 경로')
    export_parser.add_argument('--format', type=str, default='onnx', help='형식')
    export_parser.add_argument('--imgsz', type=int, default=640, help='이미지 크기')

    # 데이터셋 생성 명령
    setup_parser = subparsers.add_parser('setup', help='데이터셋 구조 생성')
    setup_parser.add_argument('--dir', type=str, default='datasets/fuel_pump_defects', help='디렉토리')

    args = parser.parse_args()

    # 기본 클래스 목록
    defect_classes = [
        'scratch',       # 스크래치
        'dent',          # 찍힘
        'crack',         # 균열
        'contamination', # 이물질
        'burr',          # 버
        'missing_part',  # 부품 누락
    ]

    if args.command == 'train':
        model_path = train_yolov8(
            data_yaml=args.data,
            model_size=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            name=args.name
        )

        if model_path:
            # ONNX 내보내기
            print("\nONNX 형식으로 내보내기...")
            export_model(model_path, 'onnx', args.imgsz)

    elif args.command == 'export':
        export_model(args.model, args.format, args.imgsz)

    elif args.command == 'setup':
        create_sample_dataset_structure(args.dir, defect_classes)

        # 데이터셋 YAML 생성
        yaml_path = os.path.join(args.dir, 'dataset.yaml')
        create_dataset_yaml(args.dir, defect_classes, yaml_path)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
