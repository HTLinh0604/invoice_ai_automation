# file: main.py 

# --- I. KHAI BÁO THƯ VIỆN ---
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os, uuid, shutil, json
from typing import List
from backend import process_receipt  # Import hàm xử lý OCR từ file backend.py
import embed_model  # Import module xử lý embedding
# Import các thành phần cần thiết từ thư viện pymilvus
from pymilvus import (
    connections, FieldSchema, CollectionSchema,
    DataType, Collection, utility
)

# --- II. KHỞI TẠO ỨNG DỤNG VÀ CẤU HÌNH ---

# Khởi tạo đối tượng ứng dụng FastAPI chính
app = FastAPI()

# 1. "Mount" thư mục static: Cho phép truy cập các file trong thư mục "static" (CSS, JS, ảnh)
# thông qua đường dẫn URL "/static". Ví dụ: /static/uploads/my_image.jpg
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. Cấu hình Jinja2 Templates: Chỉ định rằng các file template HTML nằm trong thư mục "templates".
templates = Jinja2Templates(directory="templates")

# 3. Thư mục để lưu trữ ảnh do người dùng tải lên.
UPLOAD_DIR = os.path.join("static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)  # Tạo thư mục nếu chưa tồn tại

# 4. Thư mục để lưu trữ file JSON đã được xử lý (tùy chọn, không dùng trong luồng chính)
OUTPUT_DIR = "output_structured"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- III. CẤU HÌNH VÀ KHỞI TẠO MILVUS ---

# Lấy thông tin kết nối từ biến môi trường, nếu không có thì dùng giá trị mặc định.
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
# Tên của collection sẽ được tạo trong Milvus.
COLLECTION_NAME = "invoice_collection"

def init_milvus():
    """
    Hàm khởi tạo kết nối và thiết lập collection trong Milvus.
    Hàm này sẽ được chạy một lần khi ứng dụng FastAPI khởi động.
    """
    # Kết nối đến server Milvus với alias là "default".
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

    # Để đảm bảo môi trường sạch cho mỗi lần chạy (hữu ích cho việc phát triển),
    # kiểm tra nếu collection đã tồn tại thì xóa đi để tạo mới.
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    # Định nghĩa cấu trúc (schema) cho collection.
    # Mỗi bản ghi trong collection sẽ có các trường này.
    fields = [
        # Trường ID: Khóa chính, kiểu số nguyên, tự động tăng.
        FieldSchema("id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        # Trường filename: Lưu tên file gốc, kiểu chuỗi, giới hạn 512 ký tự.
        FieldSchema("filename", dtype=DataType.VARCHAR, max_length=512),
        # Trường content: Lưu toàn bộ nội dung JSON của hóa đơn dưới dạng chuỗi.
        # max_length lớn để chứa được các hóa đơn phức tạp.
        FieldSchema("content", dtype=DataType.VARCHAR, max_length=65_535),
        # Trường embedding: Lưu vector embedding của nội dung hóa đơn.
        # `dim` (số chiều) PHẢI khớp với số chiều của model embedding.
        FieldSchema("embedding", dtype=DataType.FLOAT_VECTOR, dim=embed_model.get_embedding_dim())
    ]
    # Tạo đối tượng schema từ danh sách các trường đã định nghĩa.
    schema = CollectionSchema(fields, description="Hóa đơn đã được OCR và vector hóa")
    # Tạo collection trong Milvus với tên và schema đã cho.
    coll = Collection(name=COLLECTION_NAME, schema=schema)

    # Tạo chỉ mục (index) cho trường embedding để tăng tốc độ tìm kiếm.
    index_params = {
        "index_type": "IVF_SQ8",  # Loại index phổ biến, cân bằng giữa tốc độ và độ chính xác.
        "metric_type": "L2",      # Loại thước đo khoảng cách (Euclidean L2).
        "params": {"nlist": 128}  # Số lượng cluster, ảnh hưởng đến hiệu năng.
    }
    coll.create_index("embedding", index_params)
    # Tải collection vào bộ nhớ để sẵn sàng cho việc tìm kiếm và chèn dữ liệu.
    coll.load()
    return coll

# Gọi hàm init_milvus() ngay khi ứng dụng khởi động.
# `milvus_coll` sẽ là một đối tượng collection toàn cục, sẵn sàng để sử dụng trong các endpoint.
milvus_coll = init_milvus()


# --- IV. CÁC API ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
async def get_upload(request: Request):
    """
    Endpoint gốc (GET /).
    Trả về trang web cho phép người dùng tải lên file ảnh hóa đơn.
    """
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def post_upload(request: Request, images: List[UploadFile] = File(...)):
    """
    Endpoint xử lý việc tải lên file (POST /upload).
    Nhận một hoặc nhiều file ảnh, xử lý OCR và hiển thị kết quả.
    """
    results = []
    # Lặp qua từng file ảnh được tải lên.
    for img in images:
        # Tạo một tên file duy nhất để tránh trùng lặp.
        uid = str(uuid.uuid4())
        fn = f"{uid}_{img.filename}"
        fp = os.path.join(UPLOAD_DIR, fn)
        # Lưu file ảnh vào thư mục UPLOAD_DIR.
        with open(fp, "wb") as f:
            shutil.copyfileobj(img.file, f)

        # Gọi hàm xử lý OCR từ backend để trích xuất thông tin từ ảnh.
        data = process_receipt(fp)
        # Xử lý trường hợp OCR thất bại (hàm trả về chuỗi lỗi thay vì dict).
        if not isinstance(data, dict):
            data = {"_error": data}

        # Thêm kết quả xử lý vào danh sách.
        results.append({
            "filename": fn,
            "json": data
        })

    # Trả về trang kết quả, truyền dữ liệu đã xử lý vào template.
    return templates.TemplateResponse(
        "results.html",
        {"request": request, "results": results}
    )

@app.post("/save_milvus")
async def save_milvus(invoices: List[dict]):
    """
    Endpoint để lưu dữ liệu hóa đơn đã được xử lý vào Milvus (POST /save_milvus).
    Dữ liệu được gửi từ frontend sau khi người dùng xác nhận.
    Cấu trúc đầu vào mong đợi:
    invoices = [
      {"filename":"...", "json":{...}},
      ...
    ]
    """
    filenames, contents, texts = [], [], []
    # Chuẩn bị dữ liệu để chèn hàng loạt (batch insert).
    for inv in invoices:
        filenames.append(inv["filename"])
        # Chuyển đổi dict JSON thành một chuỗi. `ensure_ascii=False` để giữ lại ký tự tiếng Việt.
        s = json.dumps(inv["json"], ensure_ascii=False)
        contents.append(s)
        texts.append(s) # Dùng chính chuỗi JSON này để tạo embedding.

    # 1. Tạo embeddings cho tất cả các văn bản cùng một lúc.
    embs = embed_model.encode_texts(texts)
    # 2. Chèn dữ liệu (filename, content, embedding) vào Milvus.
    mr = milvus_coll.insert([filenames, contents, embs])
    # 3. Flush collection để đảm bảo dữ liệu được ghi và có thể tìm kiếm ngay lập tức.
    milvus_coll.flush()
    # 4. Lấy danh sách các ID của các bản ghi vừa được chèn.
    inserted_ids = [int(pk) for pk in mr.primary_keys]
    # 5. Trả về thông báo thành công và danh sách ID.
    return JSONResponse({"message": "Thêm dữ liệu vào Milvus thành công", "ids": inserted_ids})


@app.get("/chat", response_class=HTMLResponse)
async def chat():
    """
    Endpoint (GET /chat) trả về một trang HTML đơn giản.
    Trang này chỉ chứa một nút để chuyển hướng người dùng sang ứng dụng chat Streamlit.
    """
    html = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
      <meta charset="UTF-8">
      <title>🤖 Trợ lý hóa đơn</title>
    </head>
    <body style="padding:40px; font-family:Segoe UI, sans-serif; text-align:center; background-color:#f9f9f9;">
        <h1 style="font-size: 48px; margin-bottom: 20px;">🤖 Trợ lý Hóa đơn</h1>
        <p style="font-size: 20px; margin-bottom: 30px;">
            Ứng dụng chat đang chạy trên Streamlit, bấm nút bên dưới để mở.
        </p>
        <a href="http://localhost:8501" target="_blank"
            style="display:inline-block; padding:15px 30px;
                    background:#2196f3; color:#fff; text-decoration:none;
                    border-radius:6px; font-size:20px;">
            🚀 Mở Trợ lý
        </a>
        <p style="margin-top: 30px; font-size: 18px;">
            <a href="/" style="text-decoration:none; color:#333;">⬅️ Quay lại trang chính</a>
        </p>
    </body>
    </html>
    """
    return HTMLResponse(html)