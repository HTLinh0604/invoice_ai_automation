# file: custom_tools.py

# --- I. KHAI BÁO THƯ VIỆN ---
from datetime import datetime
import pytz  # Thư viện để làm việc với các múi giờ phức tạp.
from langchain.tools import tool  # Decorator để biến một hàm Python thành một "công cụ" cho AI Agent.
from pydantic import Field  # Dùng để cung cấp mô tả chi tiết cho các tham số của công cụ.
from typing import Literal, Optional  # Các kiểu dữ liệu giúp định nghĩa tham số rõ ràng hơn cho LLM.
import json  # Thư viện để làm việc với dữ liệu định dạng JSON.

# --- II. ĐỊNH NGHĨA CÁC CÔNG CỤ (TOOLS) ---
# Mỗi hàm được đánh dấu bằng `@tool` sẽ được AI Agent "nhìn thấy" và có thể quyết định sử dụng
# khi nhận được yêu cầu phù hợp từ người dùng.

@tool
def get_vietnam_current_time() -> str:
    """(Công cụ lấy giờ) Sử dụng khi người dùng hỏi về thời gian hiện tại. Công cụ này không có tham số."""
    # Docstring (chuỗi tài liệu) phía trên rất quan trọng.
    # LLM sẽ đọc mô tả này để quyết định khi nào nên sử dụng công cụ này.
    try:
        # Tạo một đối tượng múi giờ cho 'Asia/Ho_Chi_Minh'.
        vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        # Lấy thời gian hiện tại và chuyển đổi nó sang múi giờ Việt Nam.
        now_in_vietnam = datetime.now(vietnam_tz)
        # Định dạng chuỗi thời gian trả về cho thân thiện với người dùng.
        return f"Bây giờ là {now_in_vietnam.strftime('%H:%M:%S ngày %d-%m-%Y')} theo giờ Việt Nam."
    except Exception as e:
        # Bắt lỗi và trả về thông báo nếu có sự cố xảy ra.
        return f"Lỗi khi lấy giờ Việt Nam: {e}"

@tool
def calculator(expression: str) -> str:
    """(Công cụ tính toán) Sử dụng khi người dùng yêu cầu thực hiện một phép tính toán học. Có một tham số là 'expression'."""
    try:
        # "Làm sạch" biểu thức đầu vào để xử lý các biến thể phổ biến.
        # Ví dụ: người dùng có thể nhập '2:2' thay vì '2/2' hoặc dùng dấu phẩy cho số thập phân.
        safe_expression = expression.replace(':', '/').replace(',', '.').replace('^', '**')
        
        # Sử dụng `eval` một cách an toàn.
        # `eval` là một hàm mạnh nhưng nguy hiểm nếu không được kiểm soát.
        # Bằng cách truyền các dict rỗng cho `globals` và `locals`, chúng ta ngăn `eval`
        # truy cập vào các hàm có sẵn hoặc các biến trong môi trường, tránh các lỗ hổng bảo mật.
        result = eval(safe_expression, {"__builtins__": {}}, {})
        return f"Kết quả của phép tính '{expression}' là: {result}"
    except Exception as e:
        return f"Biểu thức toán học không hợp lệ: {e}"

@tool
def get_invoice_report(
    all_documents: list, 
    report_type: Literal['count', 'summarize', 'highest_value'] = Field(..., description="Loại báo cáo cần tạo.")
) -> str:
    """(Công cụ báo cáo) CHỈ DÙNG ĐỂ TẠO BÁO CÁO TỔNG QUAN về hóa đơn. Có một tham số là 'report_type'."""
    # Tham số `all_documents` sẽ được "tiêm" vào từ bên ngoài (trong file modelchat.py).
    # `report_type` là tham số mà LLM phải cung cấp. `Literal` giúp giới hạn các lựa chọn hợp lệ.
    
    if not all_documents: return "Không có dữ liệu hóa đơn nào để tạo báo cáo."

    # --- Nhánh 1: Đếm số lượng hóa đơn ---
    if report_type == 'count':
        return f"Bạn đã tải lên tổng cộng {len(all_documents)} hóa đơn."

    # --- Nhánh 2: Tìm hóa đơn có giá trị cao nhất ---
    if report_type == 'highest_value':
        top_invoice = None; max_value = -1.0
        for doc in all_documents:
            try:
                # Mỗi `doc` là một đối tượng Document của LangChain, nội dung hóa đơn nằm trong `page_content`.
                # Nội dung này là một chuỗi JSON, cần được `json.loads` để chuyển thành dict Python.
                invoice_data = json.loads(doc.page_content)
                current_value = float(invoice_data.get("total_amount", 0))
                if current_value > max_value:
                    max_value = current_value; top_invoice = invoice_data
            except: continue # Bỏ qua nếu có lỗi khi xử lý một hóa đơn (ví dụ: JSON không hợp lệ).
        
        if top_invoice:
            receipt_id = top_invoice.get('receipt_number', 'Không có mã')
            # Định dạng số tiền cho dễ đọc, ví dụ: 1.200.000 VND.
            formatted_value = f"{max_value:,.0f} VND".replace(',', '.')
            return f"Hóa đơn có giá trị cao nhất là hóa đơn '{receipt_id}' với tổng giá trị là {formatted_value}."
        return "Không tìm thấy hóa đơn nào có thông tin giá trị."

    # --- Nhánh 3: Tóm tắt tất cả các hóa đơn ---
    if report_type == 'summarize':
        report_lines = []
        for i, doc in enumerate(all_documents):
            try:
                invoice_data = json.loads(doc.page_content)
                receipt_id = invoice_data.get("receipt_number", f"Hóa đơn không mã số {i+1}")
                item_count = len(invoice_data.get("items", []))
                total_amount = float(invoice_data.get("total_amount", 0))
                # Lấy danh sách tên các mặt hàng.
                items_list = [item.get('name', 'N/A') for item in invoice_data.get("items", [])]
                formatted_value = f"{total_amount:,.0f} VND".replace(',', '.')
                report_lines.append(f"- Hóa đơn '{receipt_id}': có {item_count} sản phẩm (gồm: {', '.join(items_list)}), tổng giá trị {formatted_value}.")
            except: continue
        
        # Tạo báo cáo cuối cùng bằng cách ghép các dòng lại với nhau.
        final_report = [f"Đây là báo cáo tóm tắt cho {len(all_documents)} hóa đơn của bạn:"]
        final_report.extend(report_lines)
        return "\n".join(final_report)

    return "Loại báo cáo không hợp lệ. Vui lòng chọn 'count', 'summarize', hoặc 'highest_value'."

@tool
def filter_invoices(
    all_documents: list,
    receipt_number: Optional[str] = None,
    total_amount: Optional[float] = None,
    item_name: Optional[str] = None
) -> str:
    """(Công cụ lọc) CHỈ DÙNG ĐỂ LỌC, TÌM KIẾM hóa đơn theo tiêu chí cụ thể. Có các tham số tùy chọn."""
    # `Optional` cho LLM biết rằng các tham số này không bắt buộc.
    
    # Kiểm tra xem có ít nhất một tiêu chí lọc được cung cấp hay không.
    if not all([receipt_number is None, total_amount is None, item_name is None]):
        matching_invoices = []
        for doc in all_documents:
            try:
                invoice_data = json.loads(doc.page_content)
                # Giả định ban đầu là hóa đơn khớp với tiêu chí.
                match = True
                # Lần lượt kiểm tra từng tiêu chí. Nếu một tiêu chí không khớp, đặt `match = False`.
                if receipt_number is not None and str(invoice_data.get("receipt_number")) != receipt_number: match = False
                if total_amount is not None and float(invoice_data.get("total_amount", 0)) != total_amount: match = False
                if item_name is not None:
                    items = invoice_data.get("items", [])
                    # Kiểm tra xem có bất kỳ mặt hàng nào trong hóa đơn chứa `item_name` không (không phân biệt hoa thường).
                    if not any(item_name.lower() in item.get("name", "").lower() for item in items): match = False
                
                # Nếu sau tất cả các kiểm tra, `match` vẫn là True, thêm hóa đơn vào danh sách kết quả.
                if match: matching_invoices.append(invoice_data)
            except: continue # Bỏ qua các hóa đơn có dữ liệu không hợp lệ.
        
        # Trả về kết quả dưới dạng chuỗi JSON nếu tìm thấy, ngược lại trả về thông báo.
        # `ensure_ascii=False` để giữ lại ký tự tiếng Việt. `indent=2` để chuỗi JSON dễ đọc hơn.
        return json.dumps(matching_invoices, indent=2, ensure_ascii=False) if matching_invoices else "Không tìm thấy hóa đơn nào khớp với tiêu chí của bạn."
    
    return "Lỗi: Bạn phải cung cấp ít nhất một tiêu chí (số hóa đơn, tổng tiền, hoặc tên mặt hàng) để lọc."