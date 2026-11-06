# run_all.py
# 한 번에 실행: 예측 계산(predict.py, 가설 1) -> 상승/비상승 분류(classify.py, 가설 3)
#              -> 연령대 분석(age_analysis.py, 가설 2) -> 화면용 data.js 생성(make_web_data.py)
# 사용: cd python && python run_all.py
# 분석 결과가 data.js 에 들어가야 하므로 make_web_data 를 맨 마지막에 실행한다.

import predict
import classify
import age_analysis
import make_web_data

predict.main()
classify.main()
age_analysis.main()
make_web_data.main()
print('완료: web/index.html 을 열어 확인하세요.')
