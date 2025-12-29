#!/usr/bin/env python3
"""
수집된 데이터를 YOLO 학습용 데이터셋으로 변환
Convert collected data to YOLO training dataset
"""

import os
import shutil
import random
import argparse
from pathlib import Path


def prepare_yolo_dataset(input_dir: str, output_dir: str,
                         train_ratio: float = 0.8,
                         val_ratio: float = 0.15,
                         test_ratio: float = 0.05):
    """
    수집된 이미지를 YOLO 데이터셋 구조로 변환
    (라벨링 후 사용)
    """

    print("\n" + "="*50)
    print("YOLO 데이터셋 준비")
    print("="*50)

    # 출력 디렉토리 구조 생성
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    # 클래스 목록 (결함 유형)
    classes = ['scratch', 'dent', 'crack', 'contamination', 'burr', 'missing_part']

    # classes.txt 생성
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        for cls in classes:
            f.write(f"{cls}\n")
    print(f"클래스 파일 생성: {len(classes)}개 클래스")

    # 이미지 파일 수집
    all_images = []
    for cls in classes + ['normal']:
        cls_dir = os.path.join(input_dir, cls)
        if os.path.exists(cls_dir):
            images = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                     if f.endswith(('.jpg', '.png', '.jpeg'))]
            all_images.extend(images)
            print(f"  {cls}: {len(images)}장")

    print(f"\n총 이미지: {len(all_images)}장")

    # 셔플 및 분할
    random.shuffle(all_images)
    n = len(all_images)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    splits = {
        'train': all_images[:train_end],
        'val': all_images[train_end:val_end],
        'test': all_images[val_end:]
    }

    # 이미지 복사
    for split_name, images in splits.items():
        print(f"\n{split_name}: {len(images)}장 복사 중...")
        for img_path in images:
            filename = os.path.basename(img_path)
            dst_path = os.path.join(output_dir, 'images', split_name, filename)
            shutil.copy2(img_path, dst_path)

            # 라벨 파일도 복사 (있으면)
            label_path = img_path.rsplit('.', 1)[0] + '.txt'
            if os.path.exists(label_path):
                label_dst = os.path.join(output_dir, 'labels', split_name,
                                        filename.rsplit('.', 1)[0] + '.txt')
                shutil.copy2(label_path, label_dst)

    # dataset.yaml 생성
    yaml_content = f"""# Fuel Pump Defect Dataset
path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: scratch
  1: dent
  2: crack
  3: contamination
  4: burr
  5: missing_part

# Number of classes
nc: 6
"""

    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)

    print("\n" + "="*50)
    print("데이터셋 준비 완료!")
    print("="*50)
    print(f"\n출력 경로: {output_dir}")
    print(f"  - images/train: {len(splits['train'])}장")
    print(f"  - images/val: {len(splits['val'])}장")
    print(f"  - images/test: {len(splits['test'])}장")
    print(f"  - dataset.yaml 생성됨")
    print(f"  - classes.txt 생성됨")
    print("\n⚠️  라벨링이 필요합니다!")
    print("   Roboflow 또는 LabelImg로 라벨링 후 학습하세요.")


def main():
    parser = argparse.ArgumentParser(description="YOLO 데이터셋 준비")
    parser.add_argument('--input', '-i', default='datasets/collected',
                       help='수집된 이미지 경로')
    parser.add_argument('--output', '-o', default='datasets/fuel_pump_defects',
                       help='YOLO 데이터셋 출력 경로')
    parser.add_argument('--train', type=float, default=0.8,
                       help='학습 데이터 비율 (기본: 0.8)')
    parser.add_argument('--val', type=float, default=0.15,
                       help='검증 데이터 비율 (기본: 0.15)')
    parser.add_argument('--test', type=float, default=0.05,
                       help='테스트 데이터 비율 (기본: 0.05)')

    args = parser.parse_args()

    prepare_yolo_dataset(
        args.input, args.output,
        args.train, args.val, args.test
    )


if __name__ == "__main__":
    main()
