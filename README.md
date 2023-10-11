# 2023_BigContest
정형데이터분야 어드밴스드리그  
![bigcon23](https://github.com/jun-suk/PJT_2023_BigContest/assets/73885257/43638b26-f2b1-447d-911d-b09c9246f169)  
[발표자료](https://github.com/jun-suk/PJT_2023_BigContest/blob/main/%5B%E1%84%87%E1%85%A1%E1%86%AF%E1%84%91%E1%85%AD%E1%84%8C%E1%85%A1%E1%84%85%E1%85%AD%5D%20%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%92%E1%85%A7%E1%86%BC%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%87%E1%85%AE%E1%86%AB%E1%84%8B%E1%85%A3_%E1%84%8B%E1%85%A5%E1%84%83%E1%85%B3%E1%84%87%E1%85%A2%E1%86%AB%E1%84%89%E1%85%B3%E1%84%83%E1%85%B3%E1%84%85%E1%85%B5%E1%84%80%E1%85%B3_%E1%84%90%E1%85%B5%E1%86%B7_%E1%84%8B%E1%85%B5%E1%84%8B%E1%85%A5%E1%84%83%E1%85%B3%E1%84%85%E1%85%B5%E1%86%B7.pdf)  

[결과보고서](https://github.com/jun-suk/PJT_2023_BigContest/blob/main/%5B%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA%E1%84%87%E1%85%A9%E1%84%80%E1%85%A9%E1%84%89%E1%85%A5%5D%20%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%92%E1%85%A7%E1%86%BC%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%87%E1%85%AE%E1%86%AB%E1%84%8B%E1%85%A3_%E1%84%8B%E1%85%A5%E1%84%83%E1%85%B3%E1%84%87%E1%85%A2%E1%86%AB%E1%84%89%E1%85%B3%E1%84%83%E1%85%B3%E1%84%85%E1%85%B5%E1%84%80%E1%85%B3_%E1%84%90%E1%85%B5%E1%86%B7_%E1%84%8B%E1%85%B5%E1%84%8B%E1%85%A5%E1%84%83%E1%85%B3%E1%84%85%E1%85%B5%E1%86%B7.pdf)  

[코드](https://github.com/jun-suk/PJT_2023_BigContest/blob/main/%5B%E1%84%8F%E1%85%A9%E1%84%83%E1%85%B3%5D%20%E1%84%8C%E1%85%A5%E1%86%BC%E1%84%92%E1%85%A7%E1%86%BC%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%E1%84%87%E1%85%AE%E1%86%AB%E1%84%8B%E1%85%A3_%E1%84%90%E1%85%B5%E1%86%B7_%E1%84%8B%E1%85%B5%E1%84%8B%E1%85%A5%E1%84%83%E1%85%B3%E1%84%85%E1%85%B5%E1%86%B7.ipynb)
<br>

## 1. 대회 주제

- 예술의전당 콘서트홀 좌석 그룹핑
- 공연별/좌석별 적정 가격을 도출할 수 있는 가격모델 제시

<br>

## 2. 대회 기간

- 참가접수: 2023년 7월 31일 ~ 9월 15일
- 제출마감일: 2023년 9월 27일

<br>

## 3. 프로젝트 목적

- 관객이 좌석 선택에 있어 중요하게 여기는 요소를 수치화하고, 이를 관객이 이해하기 쉽게 제공하여 관객 만족도 향상을 도모함.
- 좌석 선호도를 등급 배분 및 좌석등급간 가격차의 근거로 활용하여 관객 의사를 직간접적으로 반영
- 실제 예술의전당에서 활용할 수 있도록 기존 공연기획 프로세스에 바로 포함될 수 있고, 쉽게 이해가능한 모델 설계

<br>

## 4. 프로젝트 필요성

- 공연예술상품은 경험재의 성격을 띠므로 소비자가 직접 소비해보기 전에는 품질을 가늠할 수 없고, 관객의 지불용의금액에 의해 거래가 결정되는 경향이 있으나 대체로 공연 공급자의 관점에서 좌석등급과 가격이 매겨지고 있음.
- 현재 좌석등급 간 가격차이는 명확한 가치에 근거를 두고 있지 않아 관객들은 해당 가격이 합리적이라고 파단하기 어려움

<br>

## 5. 프로젝트 내용

- 공연 클러스터링
  - '예매가 빨리 된 좌석일수록 고객 선호도가 높은 좌석'이라는 가설을 세움.
  - 공연별 예매 순서를 기준으로 좌석 선호도의 분포가 비슷한 공연끼리 클러스터링함. 

- 공연 클러스터별 주요 변수 파악
  - 1)에서 만든 공연 클러스터별 좌석 가치 판단요소(좌석 시야, 가로거리, 가로 방향(무대기준 좌,우), 세로거리, 세로 방향(무대기준 앞, 뒤), 층고)이 좌석 선호도에 미치는 영향의 크기와 양, 음의 상관관계를 파악하고, 이를 수치화함. 

- 좌석등급 분배
  - 선호도 순으로 좌석의 등급을 지정하되, 등급별 좌석의 갯수는 기존 예술의전당에서 적용중인 등급별 허용좌석수를 참고함.
  - 등급별 선호도 평균의 차를 통해 등급간 가격 차이 산정

<br>

## 6. 수행 절차
1) 탐색적 데이터 분석(EDA) 및 전처리
2) 좌석선호도에 따른 클러스터링
3) 클러스터별 EDA
4) 클러스터별 좌석 가치 판단요소 분석
5) 좌석등급분배
6) 모델 시현
