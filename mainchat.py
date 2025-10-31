# file: mainchat.py

# --- I. KHAI BÁO CÁC THƯ VIỆN CẦN THIẾT ---
import os  # Tương tác với hệ điều hành, chủ yếu để đọc biến môi trường.
import streamlit as st  # Thư viện chính để xây dựng giao diện người dùng web.
from dotenv import load_dotenv  # Tải các biến môi trường từ file .env.
import re  # Thư viện cho biểu thức chính quy (Regular Expressions), dùng để xử lý văn bản.
from milvus_utils import get_milvus_retriever  # Hàm tiện ích tự định nghĩa để lấy retriever từ Milvus.
from modelchat import create_chat_agent_executor  # Hàm tự định nghĩa để tạo ra AI agent.
from pymilvus import utility, connections  # Các công cụ để tương tác trực tiếp với Milvus (kiểm tra kết nối, liệt kê collections).

# Thư viện để quản lý lịch sử trò chuyện, tích hợp với session state của Streamlit.
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import io  # Dùng để tạo một bộ đệm (buffer) trong bộ nhớ, hoạt động như một file văn bản.
import contextlib  # Cung cấp các tiện ích, ở đây là để chuyển hướng luồng đầu ra (stdout).

# --- II. CẤU HÌNH BAN ĐẦU VÀ BIẾN TOÀN CỤC ---

# Danh sách các từ khóa chào hỏi phổ biến.
GREETING_KEYWORDS = ["xin chào", "chào bạn", "hello", "hi", "chào", "helo", "alo"]

# Tạo một mẫu biểu thức chính quy (regex) để tìm kiếm các từ chào hỏi.
# Đây là một kỹ thuật tối ưu hóa quan trọng:
# - `re.escape(keyword)`: Đảm bảo các ký tự đặc biệt trong từ khóa được xử lý đúng.
# - `|`: Toán tử "OR" trong regex, khớp với bất kỳ từ nào trong danh sách.
# - `\b`: "Word Boundary" (ranh giới từ). Điều này đảm bảo chúng ta chỉ khớp với từ hoàn chỉnh.
#   Ví dụ: `\bhi\b` sẽ khớp với "hi" trong "chào hi bạn" nhưng không khớp trong từ "hiện tại".
pattern = r"\b(" + "|".join(re.escape(keyword) for keyword in GREETING_KEYWORDS) + r")\b"
# Biên dịch (compile) regex trước để tăng tốc độ tìm kiếm khi hàm được gọi nhiều lần.
GREETING_REGEX = re.compile(pattern, re.IGNORECASE)  # IGNORECASE để không phân biệt chữ hoa/thường.

# --- III. CÁC HÀM CHỨC NĂNG ---

def initialize_app():
    """
    Hàm khởi tạo các cấu hình cơ bản cho ứng dụng Streamlit.
    """
    load_dotenv()  # Tải các biến từ file .env vào môi trường.
    # Cấu hình trang web: tiêu đề, icon và layout rộng.
    st.set_page_config(page_title="Trợ lý AI", page_icon="🤖", layout="wide")

def setup_sidebar():
    """
    Hàm thiết lập và hiển thị thanh bên (sidebar) cho phép người dùng cấu hình.
    Returns:
        tuple: (tên collection đã chọn, tên model LLM đã chọn)
    """
    # Mọi thứ bên trong khối này sẽ được hiển thị trên sidebar.
    with st.sidebar:
        st.title("⚙️ Cấu hình Trợ lý")
        
        # Kết nối đến Milvus và lấy danh sách các collection có sẵn.
        try:
            # Chỉ kết nối nếu chưa có kết nối mặc định nào tồn tại.
            if not connections.has_connection("default"):
                host = os.getenv("MILVUS_HOST", "localhost")
                port = os.getenv("MILVUS_PORT", "19530")
                connections.connect("default", host=host, port=port)
            # Lấy danh sách tên các collection.
            available_collections = utility.list_collections()
        except Exception as e:
            # Nếu không kết nối được, hiển thị lỗi và trả về danh sách rỗng.
            st.error(f"Không thể kết nối Milvus: {e}")
            available_collections = []

        def clear_agent_on_change():
            """
            Hàm callback quan trọng. Khi người dùng thay đổi cấu hình (collection hoặc model),
            hàm này sẽ xóa agent cũ khỏi bộ nhớ session.
            Điều này buộc ứng dụng phải tạo lại agent mới với cấu hình mới ở lần tương tác tiếp theo.
            """
            keys_to_delete = ["agent_executor", "collection_for_agent", "model_for_agent"]
            for key in keys_to_delete:
                if key in st.session_state:
                    del st.session_state[key]

        # Tạo dropdown để người dùng chọn collection.
        selected_collection = st.selectbox(
            "Chọn một collection để làm việc:",
            options=available_collections,
            key="collection_choice",
            on_change=clear_agent_on_change  # Gọi hàm xóa agent khi lựa chọn thay đổi.
        )

        st.header("🧠 Lựa chọn Model Trả lời")
        # Tạo dropdown để người dùng chọn model LLM.
        llm_model_name = st.selectbox("Chọn LLM (Ollama):", [
            "llama3.2:latest", "mistral:latest", "qwen:latest", "gemma:7b"
        ], key="model_choice", on_change=clear_agent_on_change)

        # Nút để áp dụng thay đổi, cũng gọi hàm xóa agent.
        st.button("🚀 Áp dụng và Khởi tạo lại", on_click=clear_agent_on_change)

    return selected_collection, llm_model_name
ANSI_ESCAPE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

def main_chat_interface():
    """
    Hàm hiển thị giao diện chat chính, xử lý đầu vào của người dùng và phản hồi.
    """
    st.title("🤖 Trợ lý AI")
    
    # Khởi tạo lịch sử chat, lưu trữ trong session state của Streamlit.
    # Điều này giúp cuộc trò chuyện không bị mất khi người dùng tương tác với các widget khác.
    msgs = StreamlitChatMessageHistory(key="chat_messages")
    # Nếu chưa có tin nhắn nào, thêm một tin nhắn chào mừng từ AI.
    if len(msgs.messages) == 0:
        msgs.add_ai_message("Chào bạn nhá! Tôi là InvoiceChatAI, trợ lý AI cực kì thông minh của bạn. Tôi có thể giúp gì cho bạn hôm nay?")

    # Hiển thị tất cả các tin nhắn đã có trong lịch sử.
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    # Tạo ô nhập liệu chat ở cuối trang.
    # `:=` (walrus operator) gán giá trị của st.chat_input cho `prompt` và kiểm tra xem nó có giá trị không trong cùng một dòng.
    if prompt := st.chat_input("💬 Hỏi về dữ liệu của bạn..."):
        # Hiển thị tin nhắn của người dùng ngay lập tức.
        st.chat_message("human").write(prompt)
        msgs.add_user_message(prompt) # Lưu tin nhắn người dùng vào lịch sử.

        # --- BỘ LỌC CHÀO HỎI THÔNG MINH ---
        # Kiểm tra xem prompt có chứa từ khóa chào hỏi không bằng regex đã biên dịch.
        # Đây là một "lối đi nhanh" (fast path) để tránh gọi đến LLM tốn kém cho các câu đơn giản.
        if GREETING_REGEX.search(prompt):
            response_content = "Chào bạn nha! Tôi có thể giúp gì cho bạn hôm nay nè?"
            with st.chat_message("assistant"):
                st.write(response_content)
        else:
            # Nếu không phải câu chào, mới thực sự gọi đến AI Agent.
            with st.chat_message("assistant"):
                # Lấy agent đã được khởi tạo và lưu trong session state.
                agent_executor = st.session_state.agent_executor
                
                # Sử dụng st.expander để tạo một khu vực có thể thu gọn/mở rộng,
                # cho phép người dùng xem quá trình suy nghĩ của AI nếu muốn.
                with st.expander("🤔 Xem quá trình suy nghĩ của AI..."):
                    log_capture_buffer = io.StringIO() # Tạo một buffer để "bắt" log.
                    # Chuyển hướng tất cả output (ví dụ: print) vào buffer thay vì console.
                    with contextlib.redirect_stdout(log_capture_buffer):
                        # Gọi agent với đầu vào và lịch sử chat.
                        # Các log từ `verbose=True` của agent sẽ được ghi vào `log_capture_buffer`.
                        response = agent_executor.invoke(
                            {"input": prompt, "chat_history": msgs.messages},
                        )
                    log_output = log_capture_buffer.getvalue() # Lấy nội dung log từ buffer.
                    log_output_clean = ANSI_ESCAPE.sub('', log_output)

                    st.text(log_output_clean) # Hiển thị log dưới dạng văn bản thô.

                # Lấy nội dung câu trả lời từ kết quả của agent.
                # Dùng .get() để tránh lỗi nếu key "output" không tồn tại.
                response_content = response.get("output", "Lỗi: Không nhận được phản hồi.")
                st.write(response_content)
        
        # Lưu tin nhắn phản hồi của AI vào lịch sử.
        msgs.add_ai_message(response_content)

# --- IV. HÀM CHÍNH VÀ ĐIỂM KHỞI CHẠY ỨNG DỤNG ---

def main():
    """
    Hàm chính điều phối toàn bộ ứng dụng.
    """
    initialize_app()
    collection_name, llm_model = setup_sidebar()

    # Nếu người dùng chưa chọn collection nào, hiển thị cảnh báo và dừng lại.
    if not collection_name:
        st.warning("Vui lòng chọn một collection ở thanh bên để bắt đầu.")
        return

    # Kiểm tra xem agent đã được khởi tạo và lưu trong session state chưa.
    # Nếu chưa, tiến hành khởi tạo.
    # Logic này đảm bảo agent chỉ được tạo MỘT LẦN cho mỗi cấu hình,
    # tránh việc phải tải lại model và thiết lập lại mọi thứ sau mỗi tin nhắn.
    if "agent_executor" not in st.session_state:
        with st.spinner(f"Đang khởi tạo Trợ lý với model '{llm_model}'..."):
            # Lấy retriever từ Milvus dựa trên collection đã chọn.
            retriever = get_milvus_retriever(collection_name)
            if retriever is None:
                st.error(f"Không thể khởi tạo retriever cho '{collection_name}'.")
                return
            
            # Tạo agent executor bằng cách gọi hàm từ modelchat.py.
            # Lưu agent đã tạo vào session state để tái sử dụng.
            st.session_state.agent_executor = create_chat_agent_executor(retriever, llm_model)
            st.success("Trợ lý đã sẵn sàng!")
    
    # Sau khi đảm bảo agent đã sẵn sàng, hiển thị giao diện chat.
    main_chat_interface()

# Điểm khởi đầu của chương trình Python.
if __name__ == "__main__":
    main()
