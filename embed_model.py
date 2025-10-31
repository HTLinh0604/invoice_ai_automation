# file: embed_model.py

# --- I. KHAI BÁO THƯ VIỆN ---
# Import thư viện SentenceTransformer, đây là thư viện phổ biến và mạnh mẽ
# để làm việc với các mô hình embedding văn bản.
from sentence_transformers import SentenceTransformer

# --- II. KHỞI TẠO MODEL ---

# 1. Định nghĩa tên của mô hình embedding sẽ được sử dụng.
# Model "dangvantuan/vietnamese-document-embedding" được chọn vì nó được huấn luyện
# chuyên biệt cho việc tạo ra các vector đại diện cho văn bản tiếng Việt.
_MODEL_NAME = "dangvantuan/vietnamese-document-embedding"

# 2. Tải mô hình từ Hugging Face Hub.
# - SentenceTransformer sẽ tự động tải model về và lưu vào cache cho các lần chạy sau.
# - `trust_remote_code=True`: Một số mô hình yêu cầu cờ này để cho phép thực thi
#   code đi kèm với mô hình trên Hub. Đây là một yêu cầu bảo mật.
# - Biến `model` này sẽ được khởi tạo một lần duy nhất khi module được import,
#   giúp tiết kiệm thời gian và tài nguyên vì không phải tải lại model mỗi lần gọi hàm.
print(f"Đang tải model embedding: {_MODEL_NAME}...")
model = SentenceTransformer(_MODEL_NAME, trust_remote_code=True)
print("✅ Model embedding đã được tải xong.")


# --- III. CÁC HÀM CHỨC NĂNG ---

def encode_texts(texts: list[str]) -> list[list[float]]:
    """
    Hàm này nhận một danh sách các chuỗi văn bản và chuyển đổi chúng thành các vector embedding.

    Args:
        texts (list[str]): Một danh sách các chuỗi văn bản cần được mã hóa.
                           Ví dụ: ["tôi là sinh viên", "bạn tên là gì?"]

    Returns:
        list[list[float]]: Một danh sách chứa các vector embedding. Mỗi vector là một
                           danh sách các số thực (float).
                           Ví dụ: [[0.1, 0.2, ...], [0.4, 0.5, ...]]
    """
    # Gọi phương thức `encode` của model để thực hiện việc chuyển đổi.
    # Đây là một hoạt động tốn nhiều tài nguyên tính toán, thường được tăng tốc bởi GPU nếu có.
    embs = model.encode(
        texts,
        batch_size=8,           # Xử lý 8 câu một lúc. Điều chỉnh số này có thể ảnh hưởng
                                # đến tốc độ và lượng VRAM sử dụng.
        convert_to_numpy=True,  # Trả về kết quả dưới dạng một mảng NumPy để xử lý hiệu quả.
        show_progress_bar=False # Ẩn thanh tiến trình khi mã hóa, hữu ích khi chạy trong môi trường server.
    )
    # Kết quả `embs` là một mảng NumPy có hình dạng (số_lượng_văn_bản, số_chiều_embedding).
    # Chuyển đổi mảng NumPy này thành một danh sách lồng nhau của Python (list of lists)
    # để dễ dàng serialize (ví dụ: gửi qua API dưới dạng JSON) hoặc chèn vào Milvus.
    return embs.tolist()

def get_embedding_dim() -> int:
    """
    Hàm tiện ích để lấy ra số chiều (dimensionality) của vector embedding mà model tạo ra.
    Con số này rất quan trọng vì nó phải được khai báo chính xác khi tạo schema trong Milvus.

    Returns:
        int: Số chiều của vector embedding (ví dụ: 768).
    """
    # Gọi phương thức có sẵn của model để lấy thông tin này.
    # Việc dùng hàm này đảm bảo rằng số chiều luôn đồng bộ với model đang được tải,
    # tránh việc phải "hard-code" một con số có thể bị sai lệch trong tương lai.
    return model.get_sentence_embedding_dimension()
