# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603686
- **Họ và tên:** Lê Nguyễn Hoàng Sơn
- **SEED (in ở ô đầu notebook):** 9973

## 1. Bằng chứng

| Tệp | Có? | Ghi chú |
| --- | --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ | Đã cung cấp notebook `day01_lab(2).ipynb`, có output; cần đổi tên thành `day01_lab.ipynb` trước khi nộp. Một số ô chưa chạy thành công. |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ | Notebook in thông báo đã tạo tệp, nhưng chưa có tệp JSON đính kèm để xác nhận/nộp. |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ | Chưa cung cấp ảnh viết tay. Các giá trị tính tay trong notebook đã qua kiểm tra 5/5, nhưng không thay thế ảnh. |

**Trạng thái notebook:** SEED = 9973; thử thách #2; 14/14 bộ kiểm tra đạt. Ô báo cáo cuối ghi thiếu 3 dự đoán trong log (`dl_relu`, `emb_search`, `emb_tricky`). Các kết quả Embedding chưa được lưu vào bảng tổng hợp; các phép gọi API LLM bị giới hạn tốc độ.

## 2. Kết quả chính

Bảng dưới đây được chép từ **bảng so sánh cuối Phần 5 của notebook**; chỉ bao gồm những mô hình có số liệu thực tế được ghi nhận.

| Phương pháp | Precision | Recall | F1 test | F1 SUB 40 | p95 (ms/tin) | USD/1k tin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 1.000 | 0.837 | 0.911 | 0.947 | 0.413 | 0.000 |
| 2. BoW + MLP (deep learning) | 0.955 | 0.924 | 0.939 | 1.000 | 0.348 | 0.000 |

Các thông số hỗ trợ: BoW + LogReg huấn luyện **4.281 giây**, đạt **13,017.987 tin/giây** khi chạy cả lô; BoW + MLP huấn luyện **6.253 giây**, đạt **16,854.002 tin/giây** khi chạy cả lô. Đây là số đo trong môi trường notebook, không phải cam kết hiệu năng khi triển khai thực tế.

**Nhận xét:** BoW + MLP có F1 test cao hơn LogReg (0.939 so với 0.911), recall cao hơn (0.924 so với 0.837) và p95 đo được cũng thấp hơn (0.348 so với 0.413 ms/tin). LogReg có precision 1.000 trên tập test, tức là tại ngưỡng đang dùng không có tin ham nào bị gán nhầm thành spam trong tập test này, nhưng bỏ sót nhiều spam hơn. Không thể điền số cho Encoder + kNN, Encoder + LogReg, Encoder + MLP hoặc LLM vì không có kết quả tương ứng trong bảng đã lưu.

## 3. Dự đoán và thực tế

**Dự đoán 1 — F1 của mô hình ML.** Em ghi trong `predict("ml_f1", ...)` rằng F1 khoảng **0.6**, precision thấp hơn recall. Kết quả test thực tế của BoW + LogReg là **F1 = 0.911**, **precision = 1.000** và **recall = 0.837**. Như vậy em đánh giá thấp chất lượng mô hình và đoán ngược thứ tự precision/recall. Điều em rút ra là các từ khóa spam trong bộ dữ liệu này đủ hữu ích để LogReg chặn rất chính xác những tin bị gắn nhãn spam, nhưng ngưỡng 0.5 khiến mô hình vẫn bỏ sót một số tin spam không mang dấu hiệu mạnh.

**Dự đoán 2 — Đường loss khi learning rate lớn.** Em ghi trong `predict("ml_lr", ...)` rằng khi `lr=0.001`, loss sẽ giảm chậm; khi `lr=50`, loss sẽ giảm rất nhanh. Sau khi xem biểu đồ, em đã tự ghi nhận trong notebook rằng dự đoán `lr=0.001` đúng, còn đường loss ở `lr=50` **tăng mạnh trước khi giảm** chứ không giảm đều. Learning rate quá lớn khiến mỗi lần cập nhật trọng số có thể vượt xa vùng tối ưu và làm loss dao động; vì vậy không thể kết luận learning rate càng lớn thì học càng tốt.

**Dự đoán 3 — MLP so với LogReg.** Em ghi trong `predict("dl_curve", ...)` rằng validation loss không giảm mãi và F1 của MLP sẽ tương đương LogReg. Kết quả cho thấy validation loss thấp nhất tại **epoch 8** trong 30 epoch, phù hợp với phần dự đoán về diễn biến loss. Tuy nhiên, MLP đạt **F1 test = 0.939**, cao hơn LogReg **0.911** (chênh 0.028); F1 SUB 40 cũng cao hơn (**1.000** so với **0.947**). Điều này cho thấy lớp ẩn có thể học những tổ hợp từ hữu ích hơn, dù mức chênh trên 40 tin SUB còn quá nhỏ để kết luận ưu thế ổn định.

## 4. Phân tích 50 case khó

**Giới hạn dữ liệu:** Trong notebook, ô chạy stress test gặp lỗi **giới hạn tốc độ API**; không có bảng F1/accuracy của các mô hình trên 50 case. Vì vậy **chưa thể khẳng định mô hình nào tụt nhiều nhất hoặc nhóm nào có tỷ lệ sai cao nhất**. Các nhận xét sau là phân tích cơ chế lỗi có thể xảy ra, chưa phải kết quả đo.

Trong dự đoán `predict("hard_eval", ...)`, em cho rằng nhóm `ham_looks_spam` khó với BoW và `prompt_injection` khó với LLM. Lý do là BoW xử lý các từ rời rạc nên có thể gán nhãn spam cho tin ham chứa từ `FREE`, `WON`, `Call`; ngược lại LLM có thể bị nội dung tin nhắn cố ý hướng dẫn thay đổi quy tắc phân loại. Nhóm `natural_spam` và `manipulation` cũng đáng chú ý vì một tin nhắn lừa đảo có thể được viết rất giống lời nhắn đời thường, không mang những từ khóa phổ biến.

Hai case cụ thể có trong bộ kiểm thử:

1. **Case #2 — `ham_looks_spam`, nhãn thật: ham.** “FREE pizza in the office kitchen right now, come quick before it's gone”. Đây là thông báo đồ ăn miễn phí trong văn phòng, không phải quảng cáo lừa đảo. BoW có **nguy cơ** dự đoán sai thành spam vì gặp `FREE`, `now`, lời thúc giục `come quick`; mô hình dựa vào ngữ cảnh có thể nhận ra đây là lời nhắn nội bộ. **Chưa có dự đoán thực tế của từng mô hình trên case này.**
2. **Case #1 — `ham_looks_spam`, nhãn thật: ham.** “Congrats!!! You WON the bet lol, I owe you a beer. Call me later”. Đây là cuộc trò chuyện bạn bè về một vụ cá cược. BoW có **nguy cơ** gán nhầm spam khi từ `Congrats`, `WON`, `Call` xuất hiện cùng nhau, dù cả câu không có mục đích phát tán spam. **Chưa có dự đoán thực tế của từng mô hình trên case này.**

Ngoài 50 case có sẵn, em đã thêm **5 case tự viết** vào `HARD_EVAL` (gồm `natural_spam`, `ham_looks_spam`, `manipulation`, `talks_about_spam` và `prompt_injection`). Tổng số case trong bảng sau khi thêm là 55; tuy nhiên ô stress test vẫn cần chạy thành công để xác định số sai thực tế. Điểm test thường có thể lạc quan vì phân phối gần dữ liệu train và chứa những mẫu từ vựng quen thuộc, trong khi bộ case khó chủ động kiểm tra ngữ cảnh, cách viết né từ khóa và khả năng chống thao túng.

## 5. Thử thách cá nhân

**SEED = 9973 → Đề #2:** “ML · Giảm vocab còn 100 từ và tăng lên 5000 từ (`build_vocab(..., max_size)`). Train lại, báo F1 test. Giải thích xu hướng.”

**Cách làm:** Em xây dựng hai từ điển bằng `build_vocab(texts_tr, max_size=size)` với `size` lần lượt là 100 và 5000. Sau đó dùng `vectorize` để biến đổi train/test theo cùng một vocab, huấn luyện lại `train_logreg` với `lr=0.5`, `epochs=300`, rồi tính precision, recall và F1 trên tập test với ngưỡng dự đoán 0.5.

| Kích thước vocab | Số từ thực tế | Precision | Recall | F1 test |
| --- | ---: | ---: | ---: | ---: |
| 100 | 100 | 0.9706 | 0.7174 | 0.8250 |
| 5000 | 5000 | 1.0000 | 0.8370 | 0.9112 |

**Kết quả và giải thích:** Vocab 5000 cho F1 cao hơn vocab 100 **0.0862**. Khi chỉ giữ 100 từ phổ biến, mô hình có thể bỏ sót từ mang tín hiệu spam, dẫn đến recall thấp hơn. Vocab lớn hơn chứa được nhiều dấu hiệu hơn nên trong thí nghiệm này giúp cải thiện khả năng phát hiện spam. Tuy nhiên, số chiều tăng cũng làm tăng bộ nhớ và chi phí tính toán, có thể làm tăng nguy cơ học thuộc đặc điểm tập huấn luyện; do đó cần đánh giá thêm trên tập validation/test trước khi chọn kích thước vocab. Mô hình BoW + LogReg ban đầu với vocab 2000 đạt F1 test xấp xỉ 0.911, gần như bằng vocab 5000, cho thấy tăng vocab không nhất thiết tiếp tục mang lại cải thiện đáng kể.

## 6. Phương án cho sản phẩm thật

Em chọn **BoW + MLP** làm bộ lọc chính cho hệ thống xử lý **10 triệu SMS/ngày** với yêu cầu **p95 < 50 ms/tin**. Cơ sở là mô hình này có **F1 test = 0.939**, **F1 SUB 40 = 1.000**, **p95 = 0.348 ms/tin**, **16,854 tin/giây** khi chạy cả lô trong notebook và không phát sinh phí API trên mỗi 1.000 tin theo bảng đo. BoW + LogReg cũng đạt SLA nhưng có F1 và recall thấp hơn trong thực nghiệm này.

Lượng tin trung bình cần xử lý xấp xỉ **10,000,000 / 86,400 = 115.74 tin/giây**. Mức throughput của MLP trong thử nghiệm cao hơn lưu lượng trung bình, nhưng **chưa đủ để bảo đảm** đáp ứng peak traffic, độ trễ hàng đợi hoặc p95 end-to-end trong sản phẩm thật. Cần đo lại trên phần cứng sản xuất, kiểm thử tải giờ cao điểm, tính cả bước nhận SMS, tiền xử lý và ghi log; theo dõi độ chính xác theo thời gian khi nội dung spam thay đổi.

**Đánh đổi:** Em chấp nhận phải triển khai, giám sát và cập nhật định kỳ một mô hình học có giám sát thay vì gọi trực tiếp LLM. MLP có precision **0.955**, thấp hơn LogReg **1.000** trên tập test; trong bối cảnh tin OTP/giao dịch bị chặn nhầm là rủi ro lớn, em sẽ ưu tiên cơ chế kiểm tra thêm cho những tin chưa chắc chắn và có thể điều chỉnh ngưỡng theo yêu cầu nghiệp vụ. Có thể dùng router chuyển các trường hợp không chắc chắn sang hàng đợi kiểm tra phụ, nhưng **không đưa việc gọi LLM đồng bộ vào đường xử lý bắt buộc** trước khi chứng minh được SLA. Không thể ước lượng đáng tin cậy chi phí LLM/tháng từ notebook vì các phép gọi thật bị giới hạn tốc độ, bảng kết quả không ghi chi phí LLM.

## 7. AI log và điều đã học

| Phần | Đã hỏi AI gì | AI trả lời / hỗ trợ gì | Đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Hoàn thiện báo cáo Day 01 | Nhờ hỗ trợ lập báo cáo từ notebook và mẫu gồm 7 mục. | Tổng hợp SEED, bảng đo, các dự đoán, thử thách cá nhân và phân tích các hạn chế của kết quả. | Đối chiếu với output notebook: SEED 9973, 14/14 checks, F1 và p95 của 2 mô hình, kết quả vocab 100/5000; đánh dấu phần stress test/LLM chưa có số liệu. |
| Các lần sử dụng AI khác (nếu có) | **Cần bổ sung đúng prompt đã sử dụng.** | **Cần bổ sung nội dung phản hồi AI thực tế.** | **Cần nêu điều tự chỉnh sửa/kiểm chứng.** |

**Điều em học được:** Trước buổi học, em nghĩ mô hình càng phức tạp thì độ chính xác càng cao, nhưng qua thực nghiệm em thấy BoW + MLP đã đạt F1 tốt cùng độ trễ rất thấp, còn LLM chưa thể đánh giá đầy đủ khi API bị giới hạn. Em cũng hiểu rằng accuracy không đủ phản ánh chất lượng khi dữ liệu mất cân bằng: phải xem precision, recall, F1 và đọc các mẫu dự đoán sai. Cuối cùng, việc thay đổi kích thước vocab cho thấy cách biểu diễn dữ liệu ảnh hưởng rõ rệt đến khả năng phát hiện spam, chứ không chỉ thuật toán huấn luyện.

---

**Những việc cần hoàn thành trước khi nộp:** Đổi tên notebook thành `day01_lab.ipynb`; cung cấp tệp `lab1_report.json` được xuất từ môi trường chạy; chụp và bổ sung `hand_calc.jpg`; chạy lại các ô Embedding/LLM và stress test nếu môi trường/API cho phép, sau đó cập nhật mục 2 và 4 theo kết quả thật; bổ sung AI log đầy đủ theo lịch sử sử dụng thực tế.
