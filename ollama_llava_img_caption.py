from ollama import generate
import glob
import pandas as pd
from PIL import Image
import os
from io import BytesIO

def load_or_create_dataframe(filename):
    """CSV 파일을 로드하거나 새로운 DataFrame을 생성합니다."""
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=['image_file', 'description'])
    return df

def get_png_files(folder_path):
    """지정된 폴더에서 PNG 파일들을 가져옵니다."""
    return glob.glob(f"{folder_path}/*.png")

def process_image_with_ollama(image_file):
    """이미지를 ollama로 처리하여 설명을 생성합니다."""
    try:
        with Image.open(image_file) as img:
            with BytesIO() as buffer:
                img.save(buffer, format='PNG')
                image_bytes = buffer.getvalue()
               
                full_response = ''
                print(f"\n처리 중: {image_file}")
               
                for response in generate(
                    model='llava:13b-v1.6',
                    prompt='describe this image and make sure to include anything notable about it (include text you see in the image). and please return Korean',
                    images=[image_bytes],
                    stream=True,
                ):
                    print(response['response'], end='', flush=True)
                    full_response += response['response']
               
                print()  # 줄바꿈
                return full_response
               
    except Exception as e:
        print(f"이미지 처리 중 오류 발생 {image_file}: {e}")
        return f"Error processing image: {e}"

def main():
    """메인 함수"""
    # 기존 데이터 로드
    df = load_or_create_dataframe('image_descriptions.csv')
   
    # 이미지 폴더 확인
    if not os.path.exists("./images"):
        print("./images 폴더가 존재하지 않습니다.")
        return
   
    # PNG 파일들 가져오기
    image_files = get_png_files("./images")
   
    if not image_files:
        print("./images 폴더에 PNG 파일이 없습니다.")
        return
   
    image_files.sort()
    print(f"총 {len(image_files)}개의 PNG 파일을 찾았습니다.")
   
    # 처리할 이미지들 (처음 5개)
    images_to_process = []
    for image_file in image_files[:5]:
        if image_file not in df['image_file'].values:
            images_to_process.append(image_file)
        else:
            print(f"이미 처리됨: {image_file}")
   
    if not images_to_process:
        print("처리할 새로운 이미지가 없습니다.")
        return
   
    print(f"{len(images_to_process)}개의 새로운 이미지를 처리합니다.")
   
    # 새로운 데이터를 저장할 리스트
    new_data = []
   
    # 각 이미지 처리
    for image_file in images_to_process:
        description = process_image_with_ollama(image_file)
        new_data.append([image_file, description])
   
    # 새로운 데이터를 DataFrame에 추가
    if new_data:
        new_df = pd.DataFrame(new_data, columns=['image_file', 'description'])
        df = pd.concat([df, new_df], ignore_index=True)
       
        # CSV 파일로 저장
        df.to_csv('image_descriptions.csv', index=False)
        print(f"\n{len(new_data)}개의 새로운 설명이 image_descriptions.csv에 저장되었습니다.")

if __name__ == "__main__":
    main()