# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603705
- **Họ và tên:** Lê Hồng Quân
- **SEED** (in ở ô đầu notebook): 5776

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | Yes |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | Yes |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | Yes |

Ghi chú: thư mục còn một bản trùng `hand_calc 2.jpg`. Nên xoá bản thừa trước khi nộp để tránh nhầm.

## 2. Kết quả chính

| Model | F1 test | F1 SUB 40 | F1 50 case khó | p95 ms/tin | USD/1k tin |
| --- | --- | --- | --- | --- | --- |
| 1. BoW + LogReg | 0.860 | 0.889 | 0.250 | 0.13 | 0 |
| 2. BoW + MLP | 0.906 | 0.947 | 0.318 | 0.36 | 0 |
| 3a. Encoder + kNN | 0.888 | 0.923 | 0.678 | 59.16 | 0 |
| 3b. Encoder + LogReg | 0.933 | 0.919 | 0.607 | 53.38 | 0 |
| 3c. Encoder + MLP head | 0.933 | 0.919 | 0.582 | 52.88 | 0 |
| 4a. LLM nhỏ zero-shot (`claude-haiku-5-5`) | — | 0.947 | 0.955 | 878.00 | 0.006 |
| 4b. LLM lớn zero-shot (`claude-opus-5-5`) | — | 0.974 | — | 1649.30 | 0.239 |
| 4c. LLM nhỏ few-shot | — | 0.974 | — | 817.87 | 0.015 |
| 4d. LLM nhỏ + prompt phòng thủ | — | 0.947 | 0.971 | — | 0.006 |

## 3. Dự đoán và thực tế

Ba dự đoán lệch nhiều nhất so với kết quả:

**(a) Thí nghiệm 2.B — bỏ ReLU (`dl_relu`)**

- *Dự đoán:* mạng không ReLU tương đương logistic regression, F1 gần bằng 0.860 của hàng 1.
- *Kết quả:* F1 test = **0.906**, bằng đúng MLP có ReLU.
- *Vì sao lệch:* lý thuyết đúng (không có hàm kích hoạt thì hai lớp Linear liên tiếp vẫn là một mô hình tuyến tính). Nhưng so sánh với hàng 1 là so sánh **hai cách train khác nhau**: LogReg dùng gradient descent thủ công, lr = 0.5, 300 epoch; mạng trong 2.B dùng Adam, lr = 1e-3, 30 epoch. Phần chênh lệch 0.86 → 0.906 đến từ cách train, không phải từ ReLU. 

**(b) Thí nghiệm 4.A — chênh lệch LLM lớn và nhỏ (`llm_models`)**

- *Dự đoán:* model lớn có F1 cao hơn "khá nhiều", và đắt gấp khoảng 50 lần.
- *Kết quả:* F1 0.974 so với 0.947 (chênh 0.027, tức đúng **1 tin trên 40**). Chi phí 0.239 so với 0.006 USD/1k tin, tức khoảng **40 lần**.
- *Vì sao lệch:* tôi đọc chênh lệch 1 tin như một khác biệt thật. Với SUB 40 tin, đó là nhiễu. Tỉ lệ giá tôi tính sai khoảng 20%.

**(c) Thí nghiệm stress test — dự đoán accuracy (`hard_eval`)**

- *Dự đoán:* BoW đạt 60–75%, LLM nhỏ đạt 75–85%.
- *Kết quả:* BoW đạt **47%** (dưới mức tôi nghĩ), LLM nhỏ đạt **95%** (trên mức tôi nghĩ).
- *Vì sao lệch:* tôi đánh giá quá cao BoW vì nó đạt 0.86–0.89 F1 trên tập test thường. Tôi đánh giá quá thấp LLM vì nghĩ nó vẫn dễ bị từ khoá đánh lừa. Thực tế là LLM đọc nghĩa câu, nên không bị "free" hay "claim" đánh lừa như BoW.

## 4. Phân tích 50 case khó

**Model nào tụt nhiều nhất.** So với F1 trên tập SUB (cùng điều kiện):

| Model | F1 SUB | F1 50 case khó | Chênh |
| --- | --- | --- | --- |
| 1. BoW + LogReg | 0.889 | 0.250 | **−0.639** |
| 2. BoW + MLP | 0.947 | 0.318 | **−0.629** |
| 3c. Encoder + MLP head | 0.919 | 0.582 | −0.337 |
| 3b. Encoder + LogReg | 0.919 | 0.607 | −0.312 |
| 3a. Encoder + kNN | 0.923 | 0.678 | −0.245 |
| 4a. LLM nhỏ zero-shot | 0.947 | 0.955 | +0.008 |
| 4d. LLM nhỏ + prompt phòng thủ | 0.947 | 0.971 | +0.024 |

- **BoW tụt mạnh nhất.** BoW + LogReg và BoW + MLP tụt gần bằng nhau, nên việc thêm lớp ẩn không giúp BoW chống được bẫy.
- **Tập test thường quá lạc quan** vì các tin trong test có cùng phân phối với train, và có nhiều tin gần như trùng nhau. Bộ 50 case khó được viết tay để tránh điều này.
- **Lưu ý cỡ mẫu:** một case đổi accuracy khoảng 2 điểm phần trăm, nên các chênh lệch nhỏ (ví dụ +0.008) không có ý nghĩa.

**Hai case cụ thể và cơ chế gây lỗi:**

1. **Case #9** (ham, nhóm "tưởng spam nhưng không phải"): *"Txt me ur address and I'll send the free tickets my boss gave me for the concert"*. Mô hình 4d (LLM nhỏ + prompt phòng thủ) gán spam. Cơ chế: câu yêu cầu gửi địa chỉ qua tin nhắn cộng với "free" khớp với mẫu lừa đảo. Mô hình thiếu ngữ cảnh "my boss gave me", vốn là dấu hiệu của người quen.

2. **Case #10** (ham, nhóm "tưởng spam nhưng không phải"): *"Call 07700 xxxxxx to reach me, my old phone died. New number!"*. Mô hình 4d gán spam. Cơ chế: cụm "New number" cộng với số điện thoại và lời kêu gọi gọi lại là mẫu quen thuộc của tin lừa đổi số. Mô hình bắt tín hiệu bề mặt mà không hiểu lý do thay số.

Với BoW, tôi dự đoán lỗi nhiều nhất ở nhóm "ham có từ khoá spam" (ví dụ case #2: *"FREE pizza in the office kitchen…"*). Lý do: "free" nằm trong nhóm từ đẩy mạnh về spam ở Phần 1, nên trọng số của nó lấn át các từ khác. 

## 5. Thử thách cá nhân

- **Đề của tôi:** Đề #5 (Embedding) — `SEED % 6 = 4`, tức chỉ số 4 trong danh sách 6 đề. Nội dung: viết 4 tin nhắn riêng (2 spam né từ khoá, 2 ham chứa "free"/"win"), so dự đoán BoW và Embedding + LogReg.
- **Cách làm:** viết 4 tin nhắn trong ô `my_messages` và đánh nhãn thật trong `true_labels = [1, 1, 0, 0]`. Dự đoán từng tin bằng BoW và Embedding + LogReg, rồi giải thích bằng lời.

| # | Tin nhắn | Nhãn thật | BoW | Embedding + LogReg |
| --- | --- | --- | --- | --- |
| 1 | Your account has been selected for an exclusive cash reward. Claim it today! | spam | spam ✓ | spam ✓ |
| 2 | We noticed unusual activity. Verify your details using this link to restore access. | spam | ham ✗ | spam ✓ |
| 3 | Feel free to call me when you arrive at the office. | ham | spam ✗ | ham ✓ |
| 4 | I hope we win the football match this weekend! | ham | spam ✗ | ham ✓ |
| | **Tổng đúng** | | **1/4** | **4/4** |

- **Giải thích:** BoW chỉ nhìn các từ rời rạc. "free" và "win" là từ mạnh về spam nên kéo tin ham về spam (tin 3, 4). Tin 2 không có từ khoá kinh điển ("verify", "link" không nằm trong nhóm từ mạnh), nên BoW đoán ham. Embedding hiểu cả câu theo nghĩa: "unusual activity" + "verify… restore access" là mẫu lừa đảo, và "feel free to call" là câu bình thường.
- **Giới hạn:** chỉ 4 câu, do tôi tự viết nên có thiên hướng.

## 6. Phương án cho sản phẩm thật

**Yêu cầu:** 10 triệu SMS/ngày, p95 < 50 ms/tin.

- Tải trung bình: 10.000.000 / 86.400 ≈ **116 tin/giây**. Tải đỉnh thực tế cao hơn nhiều lần; con số đỉnh cần đo từ dữ liệu của nhà mạng, nên tôi không đưa ra con số cụ thể.

**Loại các phương án không đạt độ trễ:**

- **LLM (4a–4c):** p95 từ 818 đến 1649 ms, gấp hơn 15 lần ngưỡng. Không thể dùng làm đường xử lý đồng bộ cho mọi tin.
- **Embedding (3a–3c):** p95 từ 53 đến 59 ms, vượt ngưỡng dù chỉ một chút. Có thể tối ưu bằng batch hoặc GPU, nhưng khi đó phải tính thêm chi phí hạ tầng mà bảng này chưa có.

**Chọn: BoW + MLP (hàng 2) làm đường xử lý chính.**

- p95 **0.36 ms**, F1 test **0.906**, đạt yêu cầu độ trễ với biên độ rất lớn.
- Tốt hơn BoW + LogReg (hàng 1) về F1 mà vẫn đủ nhanh. Với 10 triệu tin/ngày, mô hình này chạy trên CPU thường là đủ, không cần GPU.

**Thiết kế lai (router):** BoW + MLP xử lý các tin có xác suất rõ ràng (p < 0.3 hoặc p > 0.7). Chỉ các tin lưỡng lự chuyển sang LLM nhỏ (few-shot, F1 SUB 0.974).

- **Điều kiện để giữ p95 < 50 ms:** p95 tính trên toàn luồng. Chỉ khi tỉ lệ tin đi sang LLM **dưới 5%** thì p95 mới không bị kéo lên bởi nhánh chậm. Vì vậy cần đo tỉ lệ tin rơi vào dải 0.3–0.7 trên dữ liệu thật, và nhánh LLM nên chạy bất đồng bộ (không chặn đường gửi SMS).
- **Chi phí ước tính** (theo đơn giá đo được ở bảng trên):
  - Gọi LLM cho mọi tin, dùng `claude-haiku-5-5`: 10.000 × 0.006 ≈ **60 USD/ngày ≈ 1.800 USD/tháng**.
  - Gọi LLM cho mọi tin, dùng `claude-opus-5-5`: 10.000 × 0.239 ≈ **2.390 USD/ngày ≈ 71.700 USD/tháng**. Không khả thi.
  - Router, 5% tin đi LLM (few-shot, 0.015 USD/1k): 500.000 tin × 0.015/1000 ≈ **7,5 USD/ngày ≈ 225 USD/tháng**.

**Đánh đổi chấp nhận:**

1. BoW + MLP vẫn yếu trên bộ khó (F1 0.318). Spam giọng tự nhiên và né từ khoá sẽ lọt. Cần dữ liệu có nhãn mới và retrain định kỳ.
2. Nhánh LLM có độ trễ hàng trăm ms và chi phí biến đổi theo tải.
3. SMS có thể chứa thông tin cá nhân. Gửi nội dung tin lên API bên thứ ba là vấn đề pháp lý và bảo mật, cần được xem xét trước khi triển khai.
4. Các số liệu đo trên Colab CPU và tập 40 tin. Cần triển khai shadow mode trên dữ liệu thật trước khi bật cho người dùng.

## 7. AI log và điều đã học

**AI log**

Phần 1–3 là vùng không dùng AI viết code. Phần 4 được dùng AI nhưng phải ghi lại. Phần này cần bạn điền từ lịch sử chat của mình, vì notebook không ghi lại việc dùng AI.

| Lần | Hỏi AI gì | Nhận được gì | Tôi đã sửa gì |
| --- | --- | --- | --- |
| 1 | Viết TODO 4.1 `call_openai` | Bản OpenAI, sau đó bản dùng SDK `anthropic` khi đổi provider sang Claude | Đổi `LLM_PROVIDER` sang `anthropic`. Ô kiểm tra 69 mock `openai` gây lỗi 404, nên tôi viết lại mock cho `anthropic`. |
| 2 | Viết TODO 4.2 (`build_classification_prompt`) và 4.3 (`parse_label`) | Code hai hàm và kiểm tra thử | Tôi đã sửa |
| 3 | Viết TODO 4.4 (`llm_classify_batch`) | Code hàm, kiểm tra bằng `call_llm` giả | Tôi đã sửa |

**Điều tôi hiểu ra hôm nay**

 Accuracy có thể gây hiểu lầm khi dữ liệu lệch: một mô hình luôn đoán ham cũng đạt 87%. Điểm trên tập test thường chưa đủ, phải thử với các case được thiết kế để đánh lừa; với BoW, F1 test 0.86–0.89 tụt xuống 0.25 trên bộ khó. Và cách mô hình nhìn văn bản quyết định nó lỗi ở đâu: BoW đếm từ, Embedding và LLM đọc nghĩa.
