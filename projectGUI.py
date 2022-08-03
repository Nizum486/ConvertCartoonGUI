from tkinter import *
from tkinter.ttk import Labelframe
import numpy as np
from PIL import Image, ImageTk
from tkinter import filedialog
import tkinter.messagebox as msgbox
import os

import cv2
import matplotlib.image as matim

files = ""
folder_selceted = ""
outimg = Image.open("images/parrot_example.jpg")
apply_def = ""

info = """ 카툰이미지, 펜슬 이미지 변경 프로그램입니다.
1. 화면에 현재 나와있는 이미지는 예시 이미지입니다.
2. "이미지 선택" 을 통해 이미지를 선택하고 "적용" 버튼을 눌러 원본 이미지를 띄웁니다.
3. 적용 함수에서 적용시킬 필터를 선택하면 결과 이미지에 출력됩니다.
4. 저장 경로를 선택 후 저장 할 수 있습니다
"""

# 이미지 경로 탐색 함수
def browse_dest_path():
    global files
    files = filedialog.askopenfilenames(title="이미지 파일을 선택하세요,",  # 파일 대화 상자를 염
        filetypes=(("jpg 파일", "*.jpg"), ("모든 파일", "*.*")),            # 기본 파일 형식을 jpg로 두고, 옵션으로 *.* 모든 파일 선택
        initialdir="C:/")                                                  # 최초에 C:/ 경로를 보여줌    
    txt_dest_path.delete(0, END)        # txt_dest_path 에 있던 문자를 지움
    txt_dest_path.insert(0, files)      # txt_dset_path 에 선택 파일 경로 입력

# 이미지 저장 경로 탐색 함수
def browse_dest_path_2():
    global folder_selceted
    folder_selected = filedialog.askdirectory()
    if folder_selected == '':
        return
    txt_dest_path_2.delete(0, END)
    txt_dest_path_2.insert(0, folder_selected)

# 지정된 이미지로 적용시키는 함수
def apply():

    if no_img_warning() == 0: return

    global img
    global ori_img
    global ori_canvas_view

    file = txt_dest_path.get()      # txt_dest_path에 있는 문자를 읽어옴
    print(file)
    img = Image.open(file)          # 해당 파일을 오픈
    img = img.resize((450, 340))    # 이미지 크기를 450,340 크기로 변경
    ori_img = ImageTk.PhotoImage(image=img) # 이미지를 Tkinter 이미지로 변경
    ori_canvas_view = ori_canvas.create_image(0, 0, anchor = "nw", image = ori_img)

# 가우시안 블러 이미지
def Gaussian_Blur():

    if no_img_warning() == 0: return
    
    global outimg
    global output_img           # 결과물 이미지 조정을 위해 output_img를 글로벌 가져옴
    global output_canvas_view   # 결과물 이미지 출력을 위해 output_canvas_view를 글로벌로 가져옴
    global apply_def
    apply_def = "Gaussian_Blur_"

    file = txt_dest_path.get()  # 파일 저장 장소를 가져오기 위해 txt_dest_path의 텍스트를 가져옴
    img = matim.imread(file)    # imread를 통해 이미지를 읽어옴 (np.ndarray)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # 원본 이미지를 rgb2gray를 통해 회색조 영상으로 변환함
    gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)  # 가우시안 블러 필터를 적용하여 블러 이미지를 만듬
    outimg = Image.fromarray(gray_blur)              # 위 과정을 통해 얻은 np.ndarray를 이미지로 변경


    outimg = outimg.resize((450, 340))               # 영상 출력을 위해 이미지 사이즈 조정 (450*340)
    output_img = ImageTk.PhotoImage(image=outimg)    # 이미지를 tkinter이미지로 변경
    output_canvas_view = output_canvas.create_image(0, 0, anchor = "nw", image = output_img)    # 이미지를 canvas에 붙임
    print("gaussian blur")

# LUT(look up table)
def adjust_gamma(image, gamma = 1):
            invGamma = 1.0 / gamma 
            table = np.array([
                ((i/255)**invGamma) * 255 for i in np.arange(0, 256)
            ])
            lut_img = cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))
            return lut_img

# 펜슬 아트 이미지
def pencil_art():

    if no_img_warning() == 0: return

    global outimg
    global output_img           # 결과물 이미지 조정을 위해 output_img를 글로벌 가져옴
    global output_canvas_view   # 결과물 이미지 출력을 위해 output_canvas_view를 글로벌로 가져옴
    global apply_def
    apply_def = "Pencil_Art_"

    file = txt_dest_path.get()  # 파일 저장 장소를 가져오기 위해 txt_dest_path의 텍스트를 가져옴
    img = matim.imread(file)    # imread를 통해 이미지를 읽어옴 (np.ndarray)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 회색조 영상
    gray_blur = cv2.GaussianBlur(gray, (21, 21), 0) # 가우시안 블러 처리

    gray_blur_divide = cv2.divide(gray, gray_blur, scale = 256) # gray영상을 블러처리한 영상으로 나누고, 1이되는 값은 256을 곱함
    pencil_sketch = adjust_gamma(gray_blur_divide, gamma = 0.1) # LUT 사용을 통해 연산속도를 높임

    outimg = Image.fromarray(pencil_sketch)         # np.ndarray를 Image로 변경
    outimg = outimg.resize((450, 340))              # 영상 출력을 위해 이미지 사이즈 조정 (450*340)
    output_img = ImageTk.PhotoImage(image=outimg)   # 이미지를 tkinter이미지로 변경
    output_canvas_view = output_canvas.create_image(0, 0, anchor = "nw", image = output_img)

    print("pencil art")

# k-평균 알고리즘
def kmeans_cluster(img, k):
    # 차원 변화, np.float32 자료형으로 변환
    data = np.float32(img).reshape((-1, 3))

    # 최대 20번 반복하고 1픽셀 이하로 움직이면 종료
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1)

    ret, label, center = cv2.kmeans(data, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)           # 중심점 좌표를 받아옴
    result = center[label.flatten()]    # 각 픽셀을 K개의 군집 중심 색상으로 치환
    result = result.reshape(img.shape)  # 입력 영상과 동일한 형태로 변환
    return result

# 엣지 마스킹
def edge_mask(img, ksize, block_size):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # 이미지를 grayscale이미지로 변경
    gray_median = cv2.medianBlur(gray, ksize)       # midian필터를 이용하여 이미지 블러처리
    edges = cv2.adaptiveThreshold(gray_median, 255, # 적응 임계처리(이미지, 임계값, 
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, ksize) # thresholding계산법, threshold타입, 적용 영역 사이즈, 평균이나 가중평균 차감값)
    return edges

# 카툰풍 이미지
def cartoon_image():

    if no_img_warning() == 0: return

    global outimg
    global output_img           # output_img를 global로 받아옴
    global output_canvas_view   # output_canvas_view를 global로 받아옴
    global apply_def
    apply_def = "Cartoon_Art_"

    file = txt_dest_path.get()  # txt_dest_path에 적혀있는 문자를 가져와서 file에 저장
    img = matim.imread(file)    # img에 file의 경로에 적힌 이미지를 저장

    edge_Mask = edge_mask(img, 5, 7)        # edge_mask를 통한 엣지 마스크 생성
    cluster_img = kmeans_cluster(img, 7)    # kmeans_cluster를 통한 이미지 생성

    # 양방향 필터를 통한 잡음제거
    bilateral = cv2.bilateralFilter(cluster_img, d=7, sigmaColor=200, sigmaSpace=200)
    # 두 이미지 합치기
    cartoon = cv2.bitwise_and(bilateral, bilateral, mask = edge_Mask)

    outimg = Image.fromarray(cartoon)
    outimg = outimg.resize((450, 340))
    output_img = ImageTk.PhotoImage(image=outimg)
    output_canvas_view = output_canvas.create_image(0, 0, anchor = "nw", image = output_img)

    print("cartoon image")

# K 평균화 이미지
def k_cluster_img():

    if no_img_warning() == 0: return

    global outimg
    global output_img
    global output_canvas_view
    global apply_def
    apply_def = "K_Cluster_"

    file = txt_dest_path.get()
    img = matim.imread(file)

    cluster_img = kmeans_cluster(img, 7)

    outimg = Image.fromarray(cluster_img)
    outimg = outimg.resize((450, 340))
    output_img = ImageTk.PhotoImage(image=outimg)
    output_canvas_view = output_canvas.create_image(0, 0, anchor = "nw", image = output_img)

    print("k_mean_cluster image")

# 이미지 미지정시 보여줄 메시지박스 
def no_img_warning():
    if len(txt_dest_path.get()) == 0: 
        msgbox.showerror("지정된 이미지가 없습니다!", "지정된 이미지가 없습니다! 먼저 이미지를 선택해주세요")
        return 0
    else: pass

# 결과물 저장
def saveImg():

    if len(txt_dest_path_2.get()) == 0:
        msgbox.showerror("저장 경로 미선택", "저장 경로를 지정해주세요")
        return

    global outimg
    global apply_def
    file_name = apply_def + "Photo.jpg"
    dest_path = os.path.join(txt_dest_path_2.get(), file_name)
    outimg.save(dest_path)

    msgbox.showinfo("알림", "저장이 완료되었습니다.")

# 도움말
def howItWork():
    msgbox.showinfo("도움말", info)

root = Tk()               # root 라는 Tk 객체 생성
root.title("Project")     # gui 제목 설정

# 도움말 메뉴
menu = Menu(root)
menu_info = Menu(menu, tearoff=0)
menu_info.add_command(label="도움말", command=howItWork)
menu.add_cascade(label="도움말", menu = menu_info)

root.config(menu=menu)


img = Image.open("images/parrot_example.jpg")  # 먼저 보여줄 예시 이미지
w, h = img.size                                   # 이미지의 폭과 높이를 w, h에 저장

# 1. 왼쪽 프레임
Left_Frame = Frame(root, width=400, height=500)             # 왼쪽 프레임 생성 폭 400, 높이 500
Left_Frame.grid(row = 0, column = 0, padx = 5, pady = 5)    # 프레임의 위치를 0행 0열에 놓고, x축 y축을 5만큼 띄움

ori_img = ImageTk.PhotoImage(image=img)                     # tkinter에 이미지를 표현하기 위해 Tk PhotoImage로 변경
output_img = ImageTk.PhotoImage(image=img)                  

# 1.1. 원본 이미지 프레임
ori_frame = LabelFrame(Left_Frame, text = "원본", width = w, height = h) # 원본 이미지를 표현할 라벨 프레임 생성
ori_frame.pack()                                                         # 라벨 프레임을 GUI창에 붙임

# 원본 이미지 캔버스
ori_canvas = Canvas(ori_frame, width = w, height = h)                    # 이미지를 출력하기 위한 캔버스 생성
ori_canvas_view = ori_canvas.create_image(0, 0, anchor = "nw", image = ori_img)
ori_canvas.pack()


# 1.2. 파일 경로 선택 프레임
path_Frame = LabelFrame(Left_Frame, text = "이미지 선택")    # 파일 경로 선택을 위한 라벨 프레임 생성
path_Frame.pack(fill="x", padx = 5, pady = 5, ipady=4)                         # 경로 선택 프레임을 GUI창에 붙임

# 파일 경로
txt_dest_path = Entry(path_Frame)          # 파일 경로를 입력받기 위한 입력창
txt_dest_path.pack(side="left", fill="x", expand=True, padx = 5, pady = 5, ipady=4) # 외부 x,y 폭 5씩, 내부 y 높이 4씩 넓힘

# 적용 버튼
apply_button = Button(path_Frame, text="적용", width=10, command = apply) # 이미지 적용을 위한 버튼 생성
apply_button.pack(side = "right", padx = 5, pady = 5)

# 파일 경로 탐색 버튼
btn_dest_path = Button(path_Frame, text="찾아보기", width=10, command=browse_dest_path) # 경로 탐색을 위한 "찾아보기" 버튼 생성
btn_dest_path.pack(side="right", padx = 5, pady = 5)

# 1.3. 적용 함수 프레임
def_frame = Labelframe(Left_Frame, text="적용 함수")    # 적용시킬 함수들을 보여줄 라벨 프레임 생성
def_frame.pack(padx = 5, pady = 5, ipadx = 5)

btn1 = Button(def_frame, text="Gaussian Blur", width = 15, command=Gaussian_Blur)   # 버튼 1 => 가우시안 블러 적용 버튼
btn1.pack(side = "left", padx = 5, pady = 5)

btn2 = Button(def_frame, text="Pencil Art", width = 15, command = pencil_art)   # 버튼 2 => 펜슬 아트 풍 적용 버튼
btn2.pack(side = "left", padx = 5, pady = 5)

btn3 = Button(def_frame, text = "K-cluster", width = 15, command=k_cluster_img)     # 버튼 3 => k-평균-알고리즘 적용 버튼
btn3.pack(side = "left", padx = 5, pady = 5)

btn4 = Button(def_frame, text = "cartoon", width = 15, command=cartoon_image)   # 버튼 4 => 카툰 풍 이미지 적용 버튼
btn4.pack(side = "left", padx = 5, pady = 5)

# 2. 오른쪽 프레임
Right_Frame = Frame(root,  width=400, height=500)           # 오른쪽 프레임 생성
Right_Frame.grid(row = 0, column = 1, padx = 5, pady = 5)   # 프레임의 위치를 0행 1열에 놓고 x, y축으로 5씩 띄움

# 2.1. 결과물 프레임
output_Frame = LabelFrame(Right_Frame, text = "결과물", width=w, height=h)  # 결과물을 보여줄 라벨 프레임 생성
output_Frame.pack()

# 결과물 이미지 캔버스
output_canvas = Canvas(output_Frame, width = w, height = h)     # 결과물 이미지를 띄울 캔버스 생성
output_canvas_view = output_canvas.create_image(0, 0, anchor = "nw", image = output_img)
output_canvas.pack()


# 2.2. 이미지 저장소 선택
path_frame = LabelFrame(Right_Frame, text="저장경로")
path_frame.pack(fill="x", padx = 5, pady = 5, ipady=5)

txt_dest_path_2 = Entry(path_frame)
txt_dest_path_2.pack(side="left", fill="x", expand=True, padx = 5, pady = 5, ipady=4)
btn_dest_path_2 = Button(path_frame, text="찾아보기", width=10, command=browse_dest_path_2)
btn_dest_path_2.pack(side="right", padx = 5, pady = 5)


# 종료 버튼
quit_btn = Button(Right_Frame, text="닫기", width = 10, command=quit)   # 종료를 위한 버튼 생성 command = 종료
quit_btn.pack(side="right", padx = 5, pady = 5)

# 저장 버튼
save_btn = Button(Right_Frame, text="저장", width = 10, command = saveImg)
save_btn.pack(side="left", padx=5, pady=5)

root.mainloop() # gui 메인루프
