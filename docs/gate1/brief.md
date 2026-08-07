# Brief
Nhóm thực hiện: Testing

Ngày cập nhật: 02/08/2026

---

## 1. Tổng quan và mục tiêu

Tên dự án: Hệ Thống AI Agent Giám Sát Ngữ Cảnh & Phản Ứng Sự Cố An Ninh Khu Đô Thị

Vấn đề hiện tại:
Các khu đô thị hiện nay được có đến hàng trăm camera an ninh, nhưng hệ thống vận hành gặp nhiều rào cản:
- Quá tải nhân sự: Lực lượng bảo vệ mỏng, không thể giám sát hàng chục màn hình cùng lúc 24/7, dẫn đến bỏ lọt sự cố.
- Độ trễ xử lý: Sự cố thường chỉ được phát hiện sau khi xem lại camera.
- Báo động giả: Các camera tích hợp cảm biến chuyển động thông thường báo động liên tục khiến bảo vệ bị rối loạn và có xu hướng phớt lờ cảnh báo.

## 2. Giải pháp

Xây dựng một hệ thống phần mềm AI Agent triển khai trên Local Server. Hệ thống tự động phân tích luồng camera trực tiếp (RTSP), kết hợp Computer Vision để phát hiện sự kiện và Agent (LLM) để đánh giá ngữ cảnh nhằm loại bỏ báo động giả. Cảnh báo được gửi theo thời gian thực về Dashboard để con người ra quyết định cuối cùng.

## 3. Use case

Hệ thống tập trung vào 3 nhãn sự kiện sau:

- Phát hiện Xâm nhập: Cảnh báo người lạ đi vào khu vực cấm hoặc ngoài khung giờ quy định.
- Phát hiện Té ngã: Nhận diện người dân gặp sự cố ngã tại khu vực công cộng để gọi y tế.
- Phát hiện Đồ bỏ quên: Cảnh báo hành lý, vật thể lạ bị bỏ lại quá thời gian quy định tại sảnh.

Các ràng buộc vận hành thiết yếu:
- Quy trình HITL: Mọi cảnh báo phải được lực lượng bảo vệ xác nhận trên Dashboard trước khi kích hoạt báo động phần cứng.
- Bảo mật quyền riêng tư: Tự động làm mờ khuôn mặt cư dân trên luồng màn hình giám sát trực tiếp, dữ liệu không được truyền ra ngoài.

## 4. Stakeholder

- Lực lượng bảo vệ trực chốt: Người sử dụng Web Dashboard liên tục để theo dõi luồng camera và thao tác xác nhận cảnh báo.
- Ban Quản Lý Khu Đô Thị: Đơn vị đầu tư hệ thống (tối ưu chi phí vận hành, giảm rủi ro pháp lý/uy tín).
- Cư dân khu đô thị: Đối tượng được theo dõi của hệ thống.

## 5. Techstack

- Computer Vision: YOLOv8 (Detect đối tượng) + OpenCV (Làm mờ mặt).
- AI Agent: LangGraph & LLM (Phân tích chuỗi sự kiện/ngữ cảnh giảm báo động giả) + Vector DB (Lưu trữ bộ nhớ ngữ cảnh).
- Backend: FastAPI (Xử lý Stream, API, WebSocket) + PostgreSQL.
- Frontend: ReactJS (Dashboard, Pop-up cảnh báo HITL, Grid view camera).

## 6. Metrics đánh giá

- Hiệu năng Model: nhãn nhận diện 3 nhãn cốt lõi.
- Giảm báo động giả: Giảm false positives nhờ LLM reasoning.
- Thời gian thực: Độ trễ từ khi phát sinh sự cố trên camera đến khi hiển thị pop-up cảnh báo trên Web.