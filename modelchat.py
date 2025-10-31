# file: modelchat.py

# --- I. KHAI BÁO THƯ VIỆN ---
# Các thư viện cần thiết để xây dựng và chạy agent.

import os  # Thư viện tương tác với hệ điều hành, thường dùng để quản lý biến môi trường.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder # Các thành phần để xây dựng prompt cho chatbot.
from langchain_core.tools import tool # Decorator để biến một hàm Python thành một "công cụ" mà LLM có thể sử dụng.
from langchain.agents import create_tool_calling_agent, AgentExecutor # Các hàm chính để tạo ra agent và thực thi nó.
from langchain_ollama import ChatOllama # Lớp để tương tác với các mô hình ngôn ngữ lớn (LLM) chạy cục bộ qua Ollama.
from langchain_community.tools.tavily_search import TavilySearchResults # Công cụ tìm kiếm web tích hợp sẵn.
import json # Thư viện để làm việc với dữ liệu định dạng JSON.
from typing import Literal, Optional # Thư viện để định nghĩa kiểu dữ liệu, giúp LLM hiểu rõ hơn về các tham số của công cụ.

# Import các hàm công cụ được định nghĩa riêng trong file custom_tools.py.
# Việc tách các công cụ ra file riêng giúp mã nguồn gọn gàng và dễ quản lý.
from custom_tools import get_vietnam_current_time, calculator, get_invoice_report, filter_invoices

# --- II. HÀM TẠO AGENT ---
# Hàm này đóng gói toàn bộ logic để khởi tạo và cấu hình agent.

def create_chat_agent_executor(retriever, llm_model_name="llama3.2:latest"):
    """
    Hàm chính để tạo ra một AgentExecutor.
    AgentExecutor là một vòng lặp chạy agent, nhận đầu vào của người dùng, quyết định công cụ nào cần gọi,
    chạy công cụ đó, lấy kết quả và đưa lại cho agent để tạo ra câu trả lời cuối cùng.

    Args:
        retriever: Một đối tượng retriever (ví dụ: từ một vector store) có khả năng truy xuất các tài liệu liên quan.
                   Nó được dùng để lấy toàn bộ dữ liệu hóa đơn làm ngữ cảnh.
        llm_model_name (str): Tên của mô hình LLM sẽ được sử dụng thông qua Ollama.

    Returns:
        AgentExecutor: Một đối tượng agent đã được cấu hình và sẵn sàng để sử dụng.
    """
    # 1. Khởi tạo mô hình ngôn ngữ lớn (LLM)
    # Sử dụng ChatOllama để kết nối với LLM đang chạy cục bộ.
    # `temperature=0` để đảm bảo kết quả trả về có tính nhất quán cao, ít sáng tạo, phù hợp cho các tác vụ logic.
    llm = ChatOllama(model=llm_model_name, temperature=0)

    # 2. Lấy ngữ cảnh (Context) từ Retriever
    # Lấy tất cả các tài liệu (hóa đơn) từ retriever.
    # Truyền một chuỗi rỗng `""` để ra hiệu rằng chúng ta muốn lấy tất cả các tài liệu có liên quan.
    all_docs = retriever.get_relevant_documents("")

    # 3. Bọc (Wrap) các công cụ với ngữ cảnh
    # Mục đích của việc bọc lại là để "tiêm" (inject) biến `all_docs` vào các hàm công cụ gốc.
    # Điều này cho phép các công cụ truy cập vào dữ liệu hóa đơn mà không cần truyền `all_docs` mỗi lần gọi.
    # LLM sẽ chỉ thấy phiên bản đã được bọc này.

    @tool
    def get_invoice_report_with_context(report_type: Literal['count', 'summarize', 'highest_value']) -> str:
        """(NỘI BỘ) Tạo báo cáo tổng quan về hóa đơn (đếm, tóm tắt, tìm giá trị cao nhất). Dùng khi cần thống kê chung."""
        # Gọi hàm gốc từ custom_tools và truyền vào ngữ cảnh `all_docs`.
        return get_invoice_report.func(all_documents=all_docs, report_type=report_type)

    @tool
    def filter_invoices_with_context(receipt_number: Optional[str] = None, total_amount: Optional[float] = None, item_name: Optional[str] = None) -> str:
        """(NỘI BỘ) Lọc và tìm kiếm hóa đơn theo các tiêu chí cụ thể như số hóa đơn, tổng tiền, hoặc tên mặt hàng."""
        # Gọi hàm gốc và truyền vào ngữ cảnh.
        return filter_invoices.func(all_documents=all_docs, receipt_number=receipt_number, total_amount=total_amount, item_name=item_name)

    @tool
    def calculator_with_context(expression: str) -> str:
        """Thực hiện các phép tính toán học đơn giản. Ví dụ: '2*3+5/2'."""
        # Mặc dù hàm calculator không cần `all_docs`, việc bọc nó theo cùng một mẫu giúp mã nhất quán.
        # Tuy nhiên, trong trường hợp này, nó không thực sự cần thiết.
        return calculator.func(expression=expression)
    
    # Ghi chú: Công cụ `get_vietnam_current_time` không cần bọc lại.
    # Lý do: Nó là một công cụ độc lập, không cần truy cập vào ngữ cảnh `all_docs`.
    # Việc giữ nó ở dạng nguyên bản giúp LLM phân biệt rõ ràng giữa các công cụ cần dữ liệu nội bộ và các công cụ không cần.
    
    # Khởi tạo công cụ tìm kiếm web
    web_search_tool = TavilySearchResults(name="web_search", description="(INTERNET) Tìm kiếm thông tin trên Internet về thị trường, tin tức, kiến thức chung.")
    
    # 4. Tạo danh sách công cụ cuối cùng cho Agent
    # Đây là danh sách tất cả các "năng lực" mà agent có thể sử dụng.
    tools = [
        calculator_with_context,          # Công cụ tính toán
        get_invoice_report_with_context,  # Công cụ báo cáo hóa đơn
        filter_invoices_with_context,     # Công cụ lọc hóa đơn
        get_vietnam_current_time,         # Công cụ lấy giờ Việt Nam (dùng trực tiếp)
        web_search_tool                   # Công cụ tìm kiếm web
    ]

    # 5. Thiết kế System Prompt - "Bộ não" của Agent
    # Đây là phần quan trọng nhất, định hình tính cách, quy tắc và quy trình ra quyết định của agent.
    # Một system prompt chi tiết và nghiêm ngặt giúp agent hoạt động một cách logic và có thể dự đoán được.
    system_prompt = """
    Bạn là một trợ lý AI tên InvoiceChatAI. Bạn là một cỗ máy logic, chỉ làm theo lệnh, không suy diễn.

    **QUY TẮC VỀ CÔNG CỤ (BẮT BUỘC PHẢI TUÂN THEO):**
    - `get_vietnam_current_time`: Dùng khi hỏi về giờ. **CÔNG CỤ NÀY KHÔNG CÓ THAM SỐ.** Bạn phải gọi nó mà không có bất kỳ tham số nào.
    - `calculator_with_context`: Dùng cho phép tính. Có 1 tham số là `expression`.
    - `get_invoice_report_with_context`: Dùng cho báo cáo hóa đơn. Có 1 tham số là `report_type` ('count', 'summarize', 'highest_value').
    - `filter_invoices_with_context`: Dùng để lọc hóa đơn. Có các tham số tùy chọn (`receipt_number`, `total_amount`, `item_name`).
    - `web_search`: Dùng cho thông tin thị trường/Internet.

    **QUY TRÌNH RA QUYẾT ĐỊNH (THEO THỨ TỰ ƯU TIÊN):**

    1.  **TOÁN HỌC?** -> Nếu câu hỏi là một phép tính, hãy dùng `calculator_with_context`.
    2.  **GIỜ GIẤC?** -> Nếu câu hỏi về thời gian hiện tại, hãy dùng `get_vietnam_current_time`. **KHÔNG ĐƯỢC TRUYỀN BẤT KỲ THAM SỐ NÀO VÀO CÔNG CỤ NÀY.**
    3.  **LỌC HÓA ĐƠN CỤ THỂ?** -> Nếu câu hỏi chứa tiêu chí lọc cụ thể (số hóa đơn, số tiền, tên hàng), hãy dùng `filter_invoices_with_context`.
    4.  **BÁO CÁO HÓA ĐƠN?** -> Nếu câu hỏi mang tính thống kê, tổng hợp về hóa đơn:
        - Chứa từ "bao nhiêu", "số lượng" -> Dùng `get_invoice_report_with_context` với `report_type='count'`.
        - Chứa từ "cao nhất", "lớn nhất" -> Dùng `get_invoice_report_with_context` với `report_type='highest_value'`.
        - Các câu hỏi chung chung khác như "tóm tắt", "thông tin các hóa đơn" -> Dùng `get_invoice_report_with_context` với `report_type='summarize'`.
    5.  **CÒN LẠI?** -> Nếu câu hỏi không thuộc các trường hợp trên (ví dụ: hỏi về tin tức, thị trường, kiến thức chung), hãy dùng `web_search`. Nếu là chào hỏi đơn thuần, hãy trả lời trực tiếp.

    **QUY TẮC TRẢ LỜI:** Sau khi công cụ chạy xong và trả về kết quả, bạn **PHẢI** sử dụng kết quả đó để hình thành câu trả lời cuối cùng cho người dùng. **TUYỆT ĐỐI KHÔNG** được phớt lờ kết quả của công cụ.
    """

    # 6. Tạo mẫu Prompt hoàn chỉnh (Prompt Template)
    # Mẫu này kết hợp system prompt, lịch sử trò chuyện, đầu vào của người dùng và "bộ nhớ nháp" của agent.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),  # Những chỉ dẫn hệ thống không đổi
            MessagesPlaceholder(variable_name="chat_history"), # Nơi để chèn lịch sử cuộc trò chuyện (để agent có trí nhớ)
            ("human", "{input}"), # Đầu vào hiện tại của người dùng
            MessagesPlaceholder(variable_name="agent_scratchpad"), # Nơi agent ghi lại các bước suy nghĩ của nó (ví dụ: quyết định gọi công cụ nào, kết quả của công cụ đó là gì). Đây là phần bộ nhớ tạm thời cho mỗi lượt trả lời.
        ]
    )

    # 7. Tạo Agent
    # `create_tool_calling_agent` là một hàm của LangChain giúp tạo ra một agent có khả năng quyết định
    # nên gọi công cụ nào dựa trên prompt và danh sách các công cụ có sẵn.
    agent = create_tool_calling_agent(llm, tools, prompt)

    # 8. Tạo và trả về AgentExecutor
    # AgentExecutor chịu trách nhiệm thực thi agent. Nó nhận agent, danh sách công cụ,
    # và cờ `verbose=True` để in ra các bước suy nghĩ của agent trong quá trình chạy, rất hữu ích cho việc gỡ lỗi.
    return AgentExecutor(agent=agent, tools=tools, verbose=True)