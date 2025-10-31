# file: milvus_utils.py

# --- I. KHAI BÁO THƯ VIỆN ---
import os  # Thư viện để tương tác với hệ điều hành, dùng để lấy các biến môi trường.
import logging  # Thư viện để ghi log, giúp theo dõi và gỡ lỗi chương trình.
import asyncio  # Thư viện cho lập trình bất đồng bộ, cần thiết để xử lý event loop cho một số thư viện.
from pymilvus import connections, utility  # Các công cụ từ thư viện pymilvus để kết nối và quản lý Milvus.
from langchain_huggingface import HuggingFaceEmbeddings  # Lớp để tải và sử dụng các mô hình embedding từ Hugging Face Hub.
from langchain_milvus import Milvus  # Lớp tích hợp của LangChain để làm việc với Milvus như một vector store.

# --- II. CẤU HÌNH LOGGING ---
# Thiết lập cấu hình cơ bản cho việc ghi log.
# Mọi log từ mức INFO trở lên sẽ được hiển thị.
logging.basicConfig(level=logging.INFO)
# Lấy một đối tượng logger cụ thể cho file này.
logger = logging.getLogger(__name__)

# --- III. CÁC HÀM TIỆN ÍCH ---

def get_query_embedding_function():
    """
    Hàm này có nhiệm vụ duy nhất là tạo và trả về một đối tượng embedding function.
    Việc tách ra hàm riêng đảm bảo rằng cả lúc nạp dữ liệu và lúc truy vấn đều dùng
    CHUNG MỘT MÔ HÌNH EMBEDDING, điều này là bắt buộc để có kết quả chính xác.
    """
    # Chỉ định tên model embedding trên Hugging Face Hub.
    # Model này được chọn vì nó chuyên dụng cho văn bản tiếng Việt.
    model_name = "dangvantuan/vietnamese-document-embedding"
    logger.info(f"Sử dụng model embedding cho truy vấn: {model_name}")
    # Một số model yêu cầu cờ này để cho phép tải và chạy code từ xa.
    model_kwargs = {'trust_remote_code': True}
    # Khởi tạo và trả về đối tượng embedding.
    return HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

def get_milvus_retriever(collection_name, db_name="default"):
    """
    Hàm cốt lõi để thiết lập kết nối đến Milvus và tạo ra một đối tượng retriever.
    Retriever là thành phần mà LangChain Agent sẽ sử dụng để tìm kiếm thông tin
    liên quan từ cơ sở dữ liệu vector.

    Args:
        collection_name (str): Tên của collection trong Milvus mà chúng ta muốn làm việc.
        db_name (str): Tên của database trong Milvus (thường là 'default').

    Returns:
        LangChain Retriever object hoặc None nếu có lỗi xảy ra.
    """
    # Xử lý vấn đề về event loop của asyncio.
    # Một số môi trường (như Streamlit hoặc khi chạy code đồng bộ) không có sẵn một event loop đang chạy.
    # `asyncio.get_running_loop()` sẽ báo lỗi trong trường hợp đó.
    # Đoạn code này đảm bảo rằng luôn có một event loop để các thành phần bất đồng bộ có thể hoạt động.
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # Nếu không có loop nào đang chạy, tạo một cái mới và đặt nó làm loop hiện tại.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Lấy thông tin kết nối Milvus từ biến môi trường, nếu không có thì dùng giá trị mặc định.
    host = os.getenv("MILVUS_HOST", "localhost")
    port = os.getenv("MILVUS_PORT", "19530")
    alias = "default"  # Đặt bí danh cho kết nối để dễ dàng quản lý.

    try:
        # Logic đảm bảo kết nối sạch:
        # Nếu đã tồn tại một kết nối với alias 'default', hãy ngắt nó đi trước khi tạo kết nối mới.
        # Điều này tránh các lỗi tiềm ẩn do kết nối cũ hoặc cấu hình không nhất quán.
        if connections.has_connection(alias):
            logger.info(f"Đang ngắt kết nối cũ có alias='{alias}'...")
            connections.disconnect(alias)
            logger.info("Đã ngắt kết nối thành công.")

        # Thiết lập kết nối mới đến server Milvus.
        logger.info(f"Đang tạo kết nối mới tới Milvus: host='{host}', port='{port}', db_name='{db_name}'")
        connections.connect(alias, host=host, port=port, db_name=db_name)
        logger.info("✅ Kết nối mới đã được thiết lập.")

        # Kiểm tra xem collection mà người dùng muốn truy vấn có thực sự tồn tại không.
        # Đây là một bước xác thực quan trọng để tránh lỗi về sau.
        if not utility.has_collection(collection_name):
            logger.error(f"Lỗi: Collection '{collection_name}' không tồn tại trong DB '{db_name}'.")
            return None # Trả về None để báo hiệu lỗi.

        logger.info(f"Collection '{collection_name}' đã được xác nhận tồn tại.")
        
        # Lấy hàm embedding đã được định nghĩa ở trên.
        embedding_function = get_query_embedding_function()
        
        # Khởi tạo đối tượng Vector Store của LangChain trỏ đến Milvus.
        # Đây là bước quan trọng để LangChain "hiểu" được cấu trúc của collection trên Milvus.
        vector_store = Milvus(
            embedding_function=embedding_function,      # Hàm dùng để biến câu hỏi thành vector.
            collection_name=collection_name,            # Tên collection để tìm kiếm.
            connection_args={"host": host, "port": port, "db_name": db_name}, # Thông tin kết nối.
            vector_field="embedding",                   # Tên trường trong schema Milvus chứa vector.
            text_field="content",                       # Tên trường trong schema Milvus chứa nội dung văn bản gốc.
                                                        # -> Dòng này CỰC KỲ QUAN TRỌNG để retriever biết lấy văn bản từ đâu sau khi tìm thấy vector.
        )
        logger.info("✅ Đã tạo Milvus vector store thành công.")
        
        # Chuyển đổi vector store thành một retriever.
        # Retriever là một giao diện tìm kiếm chuyên dụng hơn.
        # `search_kwargs={'k': 3}`: Cấu hình retriever để luôn trả về 1000 kết quả phù hợp nhất.
        return vector_store.as_retriever(search_kwargs={'k': 1000})

    except Exception as e:
        # Bắt tất cả các lỗi có thể xảy ra trong quá trình (lỗi mạng, lỗi cấu hình,...)
        # `exc_info=True` sẽ ghi lại toàn bộ traceback của lỗi, rất hữu ích cho việc gỡ lỗi.
        logger.error(f"❌ Lỗi nghiêm trọng trong quá trình lấy retriever: {e}", exc_info=True)
        return None # Trả về None nếu có bất kỳ lỗi nghiêm trọng nào.