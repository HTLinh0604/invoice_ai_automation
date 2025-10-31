# file: backend.py 

# --- I. KHAI BÁO THƯ VIỆN ---
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image  # Pillow: Thư viện xử lý ảnh cơ bản (mở, resize, chuyển đổi).
from transformers import pipeline  # Thư viện của Hugging Face để dễ dàng sử dụng các mô hình AI.
import requests
import pytesseract  # Wrapper Python cho Tesseract OCR Engine.
import re
import cv2  # OpenCV: Thư viện xử lý ảnh và thị giác máy tính nâng cao.
from huggingface_hub import login
from dotenv import load_dotenv  # Tải các biến môi trường từ file .env.
import random
import os
import json
import google.generativeai as genai  # SDK của Google cho các mô hình Gemini.

# --- II. CẤU HÌNH BAN ĐẦU ---

# Tải các biến môi trường (ví dụ: API keys) từ file .env.
load_dotenv()
# Lấy token của Hugging Face để có thể tải các mô hình private hoặc có yêu cầu xác thực.
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("❌ Không tìm thấy HF_TOKEN trong file .env")

# Cấu hình đường dẫn đến file thực thi của Tesseract OCR trên Windows.
# Đây là bước bắt buộc nếu Tesseract không được thêm vào biến môi trường PATH của hệ thống.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# --- III. CÁC HÀM TIỀN XỬ LÝ ẢNH (IMAGE PREPROCESSING) ---
# Mục đích: Cải thiện chất lượng ảnh đầu vào để Tesseract OCR có thể nhận dạng văn bản chính xác hơn.

def resize_image_in_memory(input_image: Image.Image, target_dpi=300, min_physical_size_inches=4) -> Image.Image:
    """
    Thay đổi kích thước ảnh trong bộ nhớ để đạt được DPI (dots per inch) mục tiêu.
    OCR hoạt động tốt nhất với ảnh có độ phân giải khoảng 300 DPI.
    """
    im = input_image
    width_px, height_px = im.size
    # Tính toán số pixel tối thiểu cần có dựa trên DPI mục tiêu.
    min_pixels = int(min_physical_size_inches * target_dpi)
    
    # Nếu chiều nhỏ nhất của ảnh thấp hơn ngưỡng, tiến hành phóng to.
    if min(width_px, height_px) < min_pixels:
        scale_factor = min_pixels / min(width_px, height_px)
        new_size = (int(width_px * scale_factor), int(height_px * scale_factor))
        im = im.resize(new_size, Image.LANCZOS) # Dùng thuật toán LANCZOS cho kết quả resize chất lượng cao.
        print(f"🖼️ Đã thay đổi kích thước ảnh trong bộ nhớ thành {new_size[0]} × {new_size[1]} pixels")
    else:
        print("🖼️ Ảnh đủ lớn, không cần thay đổi kích thước.")
    return im

def auto_morphology(thresh: np.ndarray) -> np.ndarray:
    """
    Áp dụng các phép biến đổi hình thái học (morphological transformations) một cách tự động.
    Mục đích là để làm liền các ký tự bị đứt gãy hoặc loại bỏ các nhiễu nhỏ.
    """
    # Tính toán "mật độ" của các pixel văn bản (màu trắng) trên ảnh.
    text_pixels = cv2.countNonZero(thresh)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    density = text_pixels / total_pixels
    
    # Dựa vào mật độ để chọn kích thước kernel phù hợp.
    # Văn bản càng thưa -> kernel càng lớn để kết nối các phần ở xa nhau.
    if density > 0.10: ksize = (1, 1)
    elif density > 0.05: ksize = (3, 3)
    elif density > 0.01: ksize = (5, 5)
    else: ksize = (7, 7)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    # Dilate (giãn nở) để làm các nét chữ đậm hơn, liền lại.
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    # Erode (co lại) để trả các nét chữ về kích thước ban đầu.
    # Cặp DILATE-ERODE được gọi là phép "Closing", giúp lấp các lỗ nhỏ trong ký tự.
    closed = cv2.erode(dilated, kernel, iterations=1)
    return closed

def preprocess_pipeline(image: Image.Image) -> np.ndarray:
    """
    Pipeline hoàn chỉnh cho việc tiền xử lý một ảnh.
    """
    # 1. Resize ảnh để đảm bảo độ phân giải đủ tốt.
    resized_image = resize_image_in_memory(image)
    # 2. Chuyển từ định dạng PIL Image sang mảng NumPy của OpenCV.
    opencv_image = np.array(resized_image)
    # 3. Đảm bảo ảnh ở định dạng BGR mà OpenCV thường dùng.
    if opencv_image.ndim == 3 and opencv_image.shape[2] == 3:
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
    # 4. Chuyển sang ảnh xám (grayscale).
    gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
    # 5. Làm mờ ảnh nền để loại bỏ các biến thể ánh sáng không đồng đều.
    background = cv2.GaussianBlur(gray, (55, 55), 0)
    # 6. "Làm phẳng" ảnh bằng cách chia ảnh gốc cho ảnh nền.
    flattened = cv2.divide(gray, background, scale=255)
    # 7. Phân ngưỡng (thresholding) để biến ảnh thành ảnh nhị phân (đen-trắng).
    #    Sử dụng phương pháp OTSU để tự động tìm ngưỡng tối ưu.
    thresh = cv2.threshold(flattened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # 8. Áp dụng phép biến đổi hình thái học để làm sạch ảnh cuối cùng.
    closed = auto_morphology(thresh)
    return closed

# --- IV. TRÍCH XUẤT VÀ SỬA LỖI VĂN BẢN ---

def extract_text_from_image(image_path: str) -> str:
    """
    Hàm trích xuất văn bản thô từ một file ảnh.
    """
    img = Image.open(image_path)
    # Áp dụng pipeline tiền xử lý để có ảnh chất lượng tốt nhất cho OCR.
    processed_img = preprocess_pipeline(img)
    # Gọi Tesseract để thực hiện nhận dạng ký tự quang học, chỉ định ngôn ngữ là tiếng Việt.
    text = pytesseract.image_to_string(processed_img, lang='vie')
    return text

# Kiểm tra xem có GPU (CUDA) không để tăng tốc các mô hình AI.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⏳ Đang tải mô hình sửa lỗi văn bản...")
# Tải pipeline sửa lỗi chính tả tiếng Việt từ Hugging Face.
corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction-v2")
MAX_LENGTH = 512 # Tăng giới hạn để xử lý các hóa đơn dài.1024
print("👍 Mô hình sửa lỗi văn bản đã sẵn sàng.")

def correct_text(text: str) -> str:
    """
    Sửa các lỗi chính tả và lỗi OCR trong văn bản đầu vào.
    """
    # `pipeline` của transformers xử lý việc chia nhỏ văn bản dài thành các batch.
    predictions = corrector(text, max_length=MAX_LENGTH)
    return predictions[0]['generated_text']

# --- V. TRÍCH XUẤT THÔNG TIN CÓ CẤU TRÚC BẰNG LLM ---

# Lấy danh sách các API key của Gemini từ biến môi trường.
keys_str = os.getenv("GEMINI_KEYS")
if not keys_str:
    raise ValueError("❌ Không tìm thấy GEMINI_KEYS trong file .env")
# Tách chuỗi key thành một danh sách, loại bỏ các khoảng trắng thừa.
GEMINI_KEYS = [key.strip() for key in keys_str.split(",") if key.strip()]
# Chọn ngẫu nhiên một key từ danh sách để sử dụng.
# Đây là một kỹ thuật tốt để phân phối tải hoặc dùng key dự phòng nếu một key hết hạn ngạch.
selected_key = random.choice(GEMINI_KEYS)
print(f"🔐 Đang sử dụng Gemini API key: {selected_key[:5]}...")
# Cấu hình thư viện Gemini với key đã chọn.
genai.configure(api_key=selected_key)
# Khởi tạo mô hình Gemini. 'gemini-2.5-flash' là một lựa chọn tốt, cân bằng giữa tốc độ và hiệu năng.
model = genai.GenerativeModel('gemini-2.5-flash')

def extract_structured_info(text: str) -> str:
    """
    Sử dụng mô hình Gemini để chuyển đổi văn bản OCR đã sửa thành một đối tượng JSON có cấu trúc.
    """
    # Prompt là phần quan trọng nhất, nó hướng dẫn chi tiết cho LLM cách hành xử và định dạng đầu ra.
    # Một prompt chi tiết, rõ ràng và có nhiều quy tắc sẽ cho kết quả chính xác và ổn định hơn.
    prompt = f"""
Bạn là một hệ thống chuyên gia AI thông minh được huấn luyện để làm việc phân tích và OCR trích xuất dữ liệu một cách chính xác,Vai trò của bạn là một "Kế toán viên Robot", chuyên xử lý hóa đơn bán lẻ từ dữ liệu OCR thô, vốn thường không hoàn hảo có thể bị lỗi, chứa lỗi như sai chính tả, thiếu ký tự, hoặc bị mờ..
Nhiệm vụ của bạn là phân tích văn bản được cung cấp và hãy **chỉ trích xuất những thông tin thực sự có mặt rõ ràng trong nội dung**, và trả về dưới định dạng cấu trúc JSON như sau:

{{
  "store_name": string hoặc null,
  "website": string hoặc null,
  "address": string hoặc null,
  "payment_method": string hoặc null,
  "receipt_number": string hoặc null,
  "receipt_datetime": string hoặc null,
  "staff_name": string hoặc null,
  "items": [
    {{
      "name": string,
      "quantity": số hoặc null,
      "unit_price": số hoặc null,
      "total_price": số hoặc null
    }}
  ],
  "total_amount": số hoặc null,              // Tổng cộng
  "discount_amount": số hoặc null,           // Giảm giá (nếu có)
  "paid_amount": số hoặc null,               // Đã thanh toán
  "customer_paid": số hoặc null,             // Khách hàng đưa
  "change": số hoặc null                     // Tiền thừa được trả lại
}}

***QUY TRÌNH SUY LUẬN VÀ TRÍCH XUẤT:***
*Phần thông tin hóa đơn*

**Phần 1: Thông tin chung của Hóa đơn**

*   **`store_name`**:
    *   **Vị trí:** Thường nằm ở trên cùng, là dòng chữ nổi bật nhất (in hoa, cỡ chữ lớn).
    *   **Từ khóa loại trừ:** Bỏ qua các từ chung chung như "HÓA ĐƠN BÁN LẺ", "PHIẾU THANH TOÁN", "HÓA ĐƠN", "PHIẾU", "BÁN HÀNG", "BÁN LẺ", "BÁN SỈ".
    *   **Logic:** Tên cửa hàng thường đi kèm với các từ như "Công ty", "TNHH", "Cửa hàng", "Chi nhánh", "Trung tâm", "Siêu thị", "Cửa hàng tiện lợi", "TÊN ĐẠI LÝ". Nếu có nhiều tên, ưu tiên tên đầu tiên, tên in hoa, hoặc tên có dấu câu đặc biệt (ví dụ: dấu hai chấm, gạch ngang).

*   **`website`**:
    *   **Dấu hiệu:** Tìm kiếm các chuỗi văn bản chứa "www.", ".com", ".vn", ".net", hoặc "website:", "web:".
    *   **Xử lý lỗi:** OCR có thể chèn khoảng trắng (ví dụ: "www. ten cua hang .vn"). Hãy loại bỏ các khoảng trắng này để tạo thành một URL hợp lệ.

*   **`address`**:
    *   **Từ khóa:** Tìm kiếm "Địa chỉ:", "Đ/c:", "Dc:", "Địa chỉ giao hàng:", "Địa chỉ nhận hàng:", "Địa chỉ cửa hàng:".
    *   **Nội dung:** Giá trị phải chứa các thành phần của một địa chỉ (ví dụ: "Tổ", "Khu", "Phố" ,"Số", "Đường", "Phố", "Phường", "Quận", "TP",...). Nếu địa chỉ bị ngắt thành nhiều dòng, hãy ghép chúng lại.

*   **`payment_method`**:
    *   **Từ khóa:** Tìm "Hình thức thanh toán", "HTTT", "Thanh toán bằng", "Phương thức thanh toán", "Phương thức thanh toán:", "Hình thức thanh toán:", "Thanh toán:", "Phương thức thanh toán:", "Hình thức thanh toán:", "Thanh toán bằng:".
    *   **Suy luận:**
        *   Nếu thấy "Tiền mặt", "Cash", "Thanh toán tiền mặt", "TIỀN MẶT","TIEN MAT" -> "Tiền mặt".
        *   Nếu thấy "Visa", "Mastercard", "JCB", "Thẻ", "Thanh toán thẻ", "Thanh toán bằng thẻ" -> "Thẻ".
        *   Nếu thấy "Momo", "VNPay", "ZaloPay", "Thanh toán Momo", "Thanh toán VNPay", "Thanh toán ZaloPay" -> Tên của ví điện tử đó.
        *   **Logic phụ:** Nếu có trường `customer_paid` và `change`, phương thức thanh toán gần như chắc chắn là "Tiền mặt".

*   **`receipt_number`**:
    *   **Từ khóa:** Tìm "Số HĐ", "Mã GD", "Số GD", "Số hóa đơn", "Receipt No.", "Số HD", "No.","Số CT", "Mã hóa đơn", "Mã giao dịch", "Số giao dịch", "Số đơn hàng", "Số đơn hàng:", "Mã đơn hàng:", "Mã giao dịch:", "Số giao dịch:", "Số chứng từ", "Số chứng từ:", "Mã chứng từ:", "Mã chứng từ".
    *   **Đặc điểm:** Thường là một chuỗi ký tự ngắn gồm chữ và số (alphanumeric). Cần phân biệt rõ ràng với số điện thoại hoặc ngày tháng.

*   **`receipt_datetime`**:
    *   **Từ khóa:** Tìm "Ngày:", "Giờ:", "Date:", "Time:", "Thời gian", "Ngày giờ:","Ngày CT","Ngày bán", "Ngày giờ giao dịch:", "Ngày giờ thanh toán:", "Ngày giờ hóa đơn", "Ngày lập", "Ngày lập hóa đơn", "Ngày lập hóa đơn:", "Ngày giờ lập hóa đơn:", "Ngày giờ lập hóa đơn:".
    *   **Logic:** Ngày và giờ có thể nằm trên cùng một dòng hoặc hai dòng riêng biệt; hãy kết hợp chúng. Cố gắng chuẩn hóa về định dạng `YYYY-MM-DDTHH:MM:SS`. Nếu không thể, giữ nguyên chuỗi gốc.

*   **`staff_name`**:
    *   **Từ khóa:** Tìm "Thu ngân:", "Nhân viên:", "NV:", "Cashier:", "Nhân viên thu ngân:", "Nhân viên bán hàng:", "NVBH", "Nhân viên phục vụ:", "Nhân viên giao hàng:", "Nhân viên giao hàng:", "Nhân viên giao hàng:".
    *   **Đặc điểm:** Giá trị phải là tên người, không phải tên công ty hay một cụm từ chung, có thể có cả mã số nhân viên. Nếu có nhiều tên, ưu tiên tên đầu tiên.

**Phần 2: Danh sách sản phẩm (`items`)**

*   **Xác định khu vực:** Tìm vùng văn bản có cấu trúc giống bảng, thường nằm giữa thông tin cửa hàng và phần tổng tiền. Các cột thường là "Tên hàng", "SL" (Số lượng), "Đơn giá", "Thành tiền".
*   **Trích xuất từng dòng:**
    *   `name`: Là phần văn bản mô tả sản phẩm. Tên sản phẩm có thể kéo dài nhiều dòng; hãy ghép chúng lại, tên có thể không có dấu câu hoặc viết hoa, có thể có thêm chữ số hoặc ký tự đặc biệt.
    *   `quantity`: Thường là một số nguyên nhỏ (1, 2, 3...) đối với số lượng, nếu là một số thực (0.01, 0.1, 1.0, 10.0, 0.02, 0.5, 9.0, ...) đối với trọng lượng. Nếu không có, mặc định là `1`, nhưng nếu không hợp lý thì để `null`.
    *   `unit_price`: Giá của một đơn vị sản phẩm, thường là một số thực (ví dụ: 10.000, 20.500). Nếu không có giá rõ ràng, để `null`.
    *   `total_price`: Tổng tiền cho dòng đó, thường là một số thực (ví dụ: 20.000, 41.000). Nếu không có giá rõ ràng, để `null`
    *   **Quy tắc xác thực VÀNG:** Sử dụng công thức `quantity * unit_price ≈ total_price` để xác định chính xác cột nào là cột nào, ngay cả khi tiêu đề cột bị thiếu hoặc sai do OCR.

*Phần giá trị tổng tiền và thanh toán*

1.  **Phân tích ngữ nghĩa (Semantic Analysis):**
    *   Sử dụng danh sách từ khóa sau để gán nhãn cho các giá trị số:
        *   `total_amount`: "Tổng cộng", "Cộng tiền hàng", "Thành tiền", "Tổng tiền", "Tổng tiền hàng", "Tổng tiền thanh toán", "Tổng tiền hóa đơn", "Tổng tiền phải trả", "Tổng tiền thanh toán", "Tổng tiền thanh toán hóa đơn", "Tổng tiền thanh toán hóa đơn", "Tổng tiền thanh toán hóa đơn", "Cộng tiền hàng".
        *   `discount_amount`: "Giảm giá", "Chiết khấu", "Khuyến mãi", "Giảm giá tiền hàng", "Giảm giá tiền", "Giảm giá tổng tiền", "Giảm giá thanh toán", "Giảm giá hóa đơn", "Giảm giá thanh toán hóa đơn", "Giảm giá thanh toán hóa đơn", "Giảm giá thanh toán hóa đơn".
        *   `paid_amount`: "Tổng tiền thanh toán", "Khách cần trả", "Phải trả", "Tổng thanh toán", "Thanh toán", "Đã thanh toán", "Tổng tiền thanh toán", "Tổng tiền thanh toán hóa đơn", "Tổng tiền thanh toán hóa đơn", "Tổng tiền thanh toán hóa đơn", "Tổng tiền giảm giá", "Tổng tiền giảm giá hóa đơn", "Tổng".
        *   `customer_paid`: "Tiền khách đưa", "Tiền khách trả", "Tiền mặt", "Khách đưa", "Khách thanh toán", "Khách trả", "Khách thanh toán bằng tiền mặt", "Khách thanh toán bằng tiền mặt", "Khách hàng thanh toán", "Khách hàng trả tiền", "Khách hàng thanh toán bằng tiền mặt".
        *   `change`: "Tiền thối lại", "Tiền thừa", "Trả lại", "Thừa", "Tiền trả lại", "Tiền trả khách", "Tiền trả lại khách", "Tiền trả lại khách hàng", "Tiền trả lại khách hàng".
    *   Hãy linh hoạt với các biến thể do lỗi OCR (ví dụ: "Tống cọng", "Giam gia" thay vì "Tổng cộng", "Giảm giá" thay vì "Giảm giá", "Khách hàng đưa" thay vì "Khách đưa", v.v.), nếu chữ in hoa thì ưu tiên chữ in hoa, nếu chữ in thường thì ưu tiên chữ in thường.

2.  **Kiểm tra chéo bằng Logic Toán học (Logical Cross-Validation):**
    *   **Quy tắc 1:** `total_amount` - `discount_amount` phải xấp xỉ bằng `paid_amount`. Sử dụng quy tắc này để xác định `paid_amount` nếu nó không được ghi rõ.
    *   **Quy tắc 2:** `customer_paid` - `paid_amount` phải bằng `change`. Dùng quy tắc này để xác thực cả ba giá trị.
    *   **Quy tắc 3:** Nếu chỉ có một con số tổng duy nhất trên hóa đơn, nó thường là `total_amount` (và cũng là `paid_amount` nếu không có giảm giá).

3.  **Xử lý Dữ liệu không hoàn hảo (Imperfection Handling):**
    *   Đối với các con số, hãy chuẩn hóa chúng: loại bỏ ký tự không phải số (ngoại trừ dấu thập phân), diễn giải đúng các dấu phân cách hàng nghìn/thập phân.
    *   Nếu một dòng sản phẩm thiếu số lượng, mặc định là `1` nếu hợp lý. Nếu không, để `null`.

4.  **Xử lý lỗi OCR:** Hãy nhận diện và bỏ qua các lỗi phổ biến như nhầm lẫn giữa 'o' và '0', 'l' và '1', 's' và '5', các dấu chấm/phẩy trong số tiền không đúng vị trí.     
    
🔒 **Quy tắc nghiêm ngặt**:
- Không bịa, không suy luận nếu thông tin KHÔNG RÕ trong văn bản.
- Nếu thông tin không thể được xác định một cách logic hoặc không có trong văn bản, hãy đặt là `null`.
- KHÔNG tự tạo sản phẩm, tên nhân viên, tên sản phẩm, mã hóa đơn, địa chỉ hay ngày tháng nếu không có, hãy đặt là `null`.
- KHÔNG đưa ra bất kỳ giải thích, ghi chú hay văn bản nào ngoài JSON thuần.
- ƯU TIÊN sự hiện diện rõ ràng: Một giá trị được ghi rõ ràng bên cạnh từ khóa (`Tổng cộng: 50.000`) luôn được ưu tiên hơn một giá trị suy luận.
- Đảm bảo JSON đúng chuẩn để có thể `json.loads(...)` mà không lỗi.

=== Văn bản hóa đơn gốc ===
\"\"\"{text}\"\"\"
"""
    # Gửi prompt (bao gồm cả hướng dẫn và dữ liệu) đến API của Gemini.
    response = model.generate_content(prompt)
    # Trả về phần văn bản trong phản hồi của mô hình.
    return response.text

# --- VI. CÁC HÀM TIỆN ÍCH VÀ PIPELINE CHÍNH ---

def save_json_from_image_path(image_path: str, data: dict, output_root: str = "output_structured"):
    """
    Lưu dữ liệu dict vào một file JSON. Tên file JSON sẽ giống tên file ảnh.
    """
    # Lấy tên file từ đường dẫn và loại bỏ phần mở rộng (ví dụ: .jpg, .png).
    image_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Đảm bảo thư mục đầu ra tồn tại.
    os.makedirs(output_root, exist_ok=True)
    
    # Tạo đường dẫn đầy đủ đến file JSON sẽ được lưu.
    json_path = os.path.join(output_root, image_filename + ".json")
    
    # Ghi dữ liệu dict vào file JSON.
    # `ensure_ascii=False` để giữ lại các ký tự tiếng Việt.
    # `indent=2` để file JSON được định dạng đẹp, dễ đọc.
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Đã lưu dữ liệu có cấu trúc vào: {json_path}")

def process_receipt(image_path: str):
    """
    Hàm chính, điều phối toàn bộ pipeline xử lý một hóa đơn từ A đến Z.
    """
    print("\n" + "="*50)
    print(f"🚀 Bắt đầu pipeline xử lý cho: {os.path.basename(image_path)}")
    print("="*50)
    
    # Bước 1: Trích xuất văn bản thô từ ảnh bằng OCR.
    print("🔍 Đang thực hiện OCR...")
    raw_text = extract_text_from_image(image_path)
    
    # Bước 2: Sửa lỗi chính tả và lỗi OCR.
    print("🧠 Đang sửa lỗi văn bản...")
    corrected_text = correct_text(raw_text)
    
    # Bước 3: Trích xuất thông tin có cấu trúc bằng LLM.
    print("📦 Đang trích xuất các trường dữ liệu có cấu trúc...")
    structured_data_str = extract_structured_info(corrected_text)
    
    # Bước 4: Làm sạch chuỗi JSON trả về từ LLM.
    # LLM đôi khi trả về chuỗi JSON nằm trong khối mã markdown (```json ... ```).
    cleaned_struct_data = structured_data_str.strip()
    if cleaned_struct_data.startswith("```json"):
        cleaned_struct_data = cleaned_struct_data.removeprefix("```json").strip()
    if cleaned_struct_data.endswith("```"):
        cleaned_struct_data = cleaned_struct_data.removesuffix("```").strip()
        
    # Bước 5: Chuyển chuỗi JSON thành đối tượng dict của Python.
    try:
        struct_data_dict = json.loads(cleaned_struct_data)
        print("✅ Dữ liệu có cấu trúc đã được parse thành công.")
    except json.JSONDecodeError as e:
        # Nếu LLM trả về một chuỗi không phải là JSON hợp lệ, báo lỗi và trả về chuỗi thô.
        print(f"❌ Lỗi khi parse JSON: {e}")
        return cleaned_struct_data
        
    # Bước 6 (Tùy chọn): Lưu file JSON xuống đĩa.
    # print("💾 Đang lưu dữ liệu có cấu trúc...")
    # save_json_from_image_path(image_path, struct_data_dict)
    
    print("🎉 Pipeline đã hoàn tất thành công!")
    # Trả về kết quả cuối cùng là một đối tượng dict.
    return struct_data_dict