# invoice_ai_automation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?logo=streamlit)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?logo=fastapi)
![Milvus](https://img.shields.io/badge/Milvus-Vector%20DB-blueviolet?logo=milvus)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-LLM-blue?logo=google)
![LangChain](https://img.shields.io/badge/LangChain-Agent-lightgrey?logo=langchain)
![Tesseract](https://img.shields.io/badge/Tesseract-OCR-lightgrey?logo=tesseract)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)

Đây là Đồ án môn học Trí tuệ Nhân tạo (2025), xây dựng một pipeline tự động, thông minh để biến ảnh hóa đơn thành một cơ sở tri thức có tổ chức, cho phép truy vấn bằng ngôn ngữ tự nhiên.

## 👥 Thông tin dự án
* **Đề tài:** Ứng Dụng Trí Tuệ Nhân Tạo Trong Nhận Diện và Tự Động Hóa Lưu Trữ Hóa Đơn
* **Trường:** Đại học Công nghệ TP. Hồ Chí Minh (HUTECH) - Khoa Công nghệ Thông tin
* **Giảng viên hướng dẫn:** TS. Hoàng Văn Quý
* **Sinh viên thực hiện:**
    * Hồ Gia Thành
    * Huỳnh Thái Linh
    * Trương Minh Khoa

---

## 📌 Vấn đề cốt lõi (The Problem)
> Xử lý hóa đơn thủ công là một "cơn ác mộng" về sai sót, tốn kém thời gian và nhân lực. Dữ liệu hóa đơn, một nguồn tài nguyên quý giá, thường bị "lãng quên" sau khi nhập liệu, không thể khai thác cho mục đích phân tích.

## 💡 Giải pháp của chúng tôi (Our Solution)
> Chúng tôi xây dựng một pipeline tự động hóa toàn diện, chia làm 3 giao diện trực quan:
> 1.  **Giao diện "Upload":** Tự động trích xuất thông tin từ ảnh hóa đơn.
> 2.  **Giao diện "Kết Quả":** Cho phép người dùng đối chiếu, chỉnh sửa (nếu cần), và xác nhận lưu trữ vào Vector Database (Milvus).
> 3.  **Giao diện "Trợ Lý Chatbot":** Cho phép người dùng truy vấn ngữ nghĩa (hỏi bằng ngôn ngữ tự nhiên) trên toàn bộ kho lưu trữ hóa đơn.

---

## 🚀 Đặc điểm nổi bật & Đổi mới
Dự án này vượt xa các giải pháp OCR truyền thống bằng cách tích hợp một pipeline đa tầng phức tạp và tùy chỉnh sâu cho ngữ cảnh Việt Nam.

* **Tích hợp Công nghệ Đa tầng:** Xây dựng pipeline liền mạch:
    1.  **Xử lý ảnh** (OpenCV)
    2.  **OCR** (Tesseract)
    3.  **Sửa lỗi Tiếng Việt** (Mô hình `seq2seq` - `bmd1905/vietnamese-correction-v2`)
    4.  **Trích xuất cấu trúc** (LLM - Google Gemini + Prompt Engineering)
    5.  **Vector hóa & Lưu trữ** (Milvus + `dangvantuan/vietnamese-document-embedding`)
    6.  **Truy vấn (RAG)** (LangChain Agent + Streamlit)

* **Địa phương hóa Chuyên biệt cho Việt Nam:**
    * **Dữ liệu:** Huấn luyện và tinh chỉnh trên bộ dữ liệu 1150+ hóa đơn, trong đó có 150+ ảnh tự thu thập từ **Bách Hóa Xanh** để xử lý đặc thù tiếng Việt, định dạng VND.
    * **Mô hình:** Sử dụng mô hình embedding tiếng Việt chuyên biệt (`vietnamese-document-embedding`) giúp cải thiện độ chính xác 20-30% so với các mô hình generic.

* **Truy vấn Ngữ nghĩa (Semantic Search):** Thay vì tìm kiếm từ khóa (`CTRL+F`), hệ thống cho phép người dùng hỏi bằng ngôn ngữ tự nhiên (ví dụ: *"Tháng này tôi mua hàng ở Bách Hóa Xanh bao nhiêu lần?"*).

* **AI Agent thông minh:** Chatbot không chỉ tìm kiếm, mà còn là một Agent (LangChain) có khả năng hiểu ngữ cảnh, quản lý lịch sử trò chuyện, và tổng hợp thông tin từ *nhiều* hóa đơn để đưa ra câu trả lời hoàn chỉnh.

* **Kiểm soát bởi Người dùng (Human-in-the-Loop):** Giao diện "Kết Quả" cho phép người dùng đối chiếu ảnh gốc và dữ liệu JSON, đảm bảo 100% độ tin cậy trước khi lưu trữ.

---

## ⚙️ Kiến trúc hệ thống
Hệ thống bao gồm Khối xử lý OCR và Khối Chatbot AI, tương tác với nhau thông qua cơ sở dữ liệu vector Milvus.

![Sơ đồ kiến trúc hệ thống](path/to/your-system-architecture-diagram.png)

### Luồng xử lý OCR (Upload Pipeline)
1.  **Tiếp nhận:** Nhận ảnh hóa đơn (hỗ trợ nhiều ảnh).
2.  **Tiền xử lý ảnh:** Áp dụng `Gaussian Blur`, `Thresholding OTSU`, và các phép toán hình thái để làm rõ văn bản.
3.  **OCR & Sửa lỗi:** Sử dụng Tesseract (`lang='vie'`) để trích xuất văn bản thô, sau đó đưa qua mô hình `vietnamese-correction-v2` để sửa lỗi chính tả.
4.  **Trích xuất cấu trúc (LLM):** Sử dụng **Gemini** với các prompt được thiết kế kỹ (Prompt Engineering) và các quy tắc kiểm tra chéo (ví dụ: `Tổng tiền sản phẩm` = `Tổng các sản phẩm con`) để chuyển văn bản thô thành định dạng `JSON` chuẩn hóa.
5.  **Tạo Embedding & Lưu trữ:** Dữ liệu JSON được vector hóa bằng mô hình SentenceTransformer (768 chiều) và lưu trữ vào **Milvus**.

### Luồng Chatbot (Query Pipeline)
1.  **Giao diện:** Người dùng đặt câu hỏi qua **Streamlit**.
2.  **Backend:** **FastAPI** và **LangChain Agent** tiếp nhận câu hỏi.
3.  **Truy xuất (Retrieve):** Agent sử dụng Milvus retriever để tìm kiếm ngữ nghĩa các hóa đơn liên quan đến câu hỏi.
4.  **Tổng hợp (Generate):** LLM (Gemini) nhận bối cảnh (câu hỏi + lịch sử chat + thông tin hóa đơn được truy xuất) và tổng hợp thành câu trả lời bằng ngôn ngữ tự nhiên.

---

## 📊 Kết quả thực nghiệm
* **Hệ thống OCR:**
    * Đạt **độ chính xác 80-90%** với ảnh rõ nét, chuẩn bố cục.
    * Linh hoạt với nhiều loại hóa đơn (siêu thị, cà phê, nhà hàng).
    * Tốc độ xử lý: Trung bình 15-20 giây cho một bộ hóa đơn.
    * *Hạn chế:* Vẫn phụ thuộc vào chất lượng ảnh (mờ, nghiêng) và các bố cục quá đặc biệt.
* **Trợ lý ảo (Chatbot):**
    * Trả lời tốt các truy vấn dữ liệu nội bộ (mã, số lượng, tổng tiền).
    * Có khả năng tìm kiếm thông tin bên ngoài hệ thống (ví dụ: so sánh giá thị trường).
    * *Hạn chế:* Đôi khi gặp lỗi hệ thống (traceback) khi truy vấn quá phức tạp hoặc lặp lại.

![Demo Giao diện Hệ thống](path/to/your-demo-screencast.gif)

---

## 🚀 Hướng phát triển
* **Nâng cấp OCR:** Thay thế Tesseract bằng các mô hình Deep Learning (CNN, ResNet) để tăng độ chính xác với ảnh chất lượng thấp.
* **Tăng cường AI Agent:** Tích hợp thêm nhiều công cụ (tools) cho LangChain Agent.
* **Triển khai Cloud:** Đóng gói hệ thống (Docker) và triển khai trên nền tảng cloud (AWS/GCP/Azure).
* **Bảo mật:** Tăng cường cơ chế bảo mật và phân quyền truy cập.
