#!/bin/bash
# run_ann.sh

# 기존 결과 디렉토리 비우기
rm -rf results
mkdir -p results

# 실행
python ann.py | tee log.txt

# 종료 상태 확인
if [ $? -eq 0 ]; then
    echo "===== 실험 성공 ====="
    echo "결과는 'results' 디렉토리에 저장되었습니다."
    echo "전체 로그는 'log.txt'에서 확인할 수 있습니다."
    
    # 결과 확인
    echo -e "\n생성된 파일 목록:"
    ls -l results/
    
    # 시계열 차트 목록
    echo -e "\n시계열 비교 차트:"
    find results -name "time_series_*.png" | sort
    
    # 성능 비교 차트 표시
    if [ -f correct_approach_results/comparison_RMSE.png ]; then
        echo -e "\nRMSE 비교 차트가 생성되었습니다: results/comparison_RMSE.png"
    fi
    
    if [ -f correct_approach_results/comparison_R².png ]; then
        echo -e "\nR² 비교 차트가 생성되었습니다: results/comparison_R².png"
    fi
else
    echo "===== 실험 실패 ====="
    echo "오류 로그 확인:"
    tail -n 20 log.txt
fi