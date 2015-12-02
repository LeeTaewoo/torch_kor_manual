# 실습 3
기계 학습,  2015년 봄

## 코스 페이지
<https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/>

## 참고 자료
- [optim](https://github.com/LeeTaewoo/torch_kor_manual/blob/master/optim/index.md)
- [paths](https://github.com/LeeTaewoo/torch_kor_manual/tree/master/paths)
- [gnuplot](https://github.com/torch/gnuplot)

## 재료
글의 "소개" 부분에서 언급된 예제를 위해 `simple_example.lua`를 보십시오. 그리고 이전 과제의 템플릿을 위해 `practical3.lua`를 보십시오.

우리는 MNIST라 불리는 데이터세트에 있는 손으로 쓴 숫자들을 분류할 것입니다. 그 데이터의 모습은 다음과 같습니다:

![mnist](https://github.com/oxford-cs-ml-2015/practical3/raw/master/mnist.png)

우리가 가진 MNIST 버전에서 각 데이터포인트는 32x32 영상입니다. 제공된 코드는 이것을 raw 픽셀 값들로 구성된 벡터 하나로 바꿀 것입니다. `simple_example.lua`에는, 훈련/시험 세트들에서 어떻게 한 숫자를 보여주는지를 그림으로 보여주는 "UNCOMMENT(주석 해제)"라 쓰인 줄이 하나 있습니다. 

## 과제 1: 소개 및 복습 
이 예제에서 (비록 쓸모는 없지만) 설명을 위해, 우리는 다음 함수의 최적화 과정을 보입니다.

<img src="https://lh4.googleusercontent.com/aHRd_MbqGTG0XG7Hf4NoKpyJglIQw0k0jNiTQbCGiizR9QjjQVG_opztT0WoP69OYBAdLOHXVnky6gi1AU39eXOkxCjD4O4u72_GOgRREGA83cjNfHciQi5Q7Gy-tlCL0wSSwKI" width="169px;" height="37px;" style="border: none; transform: rotate(0.00rad); -webkit-transform: rotate(0.00rad);" title="This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.">

그 함수의 기울기(gradient)는 다음과 같습니다.

<img src="https://lh6.googleusercontent.com/hN29KcfEC3PKf_0GxZ5uxrAxAYbcBiALRxpuYyviw1bXvJ0IDlujgap4s1toDUCWzMdGH6IMFAcJpES3iyJ7yadDhlCpKqNRnJXYi413GADcNnqJskHYTcekQEf2hBu98VhwPvs" width="253px;" height="19px;" style="border: none; transform: rotate(0.00rad); -webkit-transform: rotate(0.00rad);" title="This is the rendered form of the equation. You can not edit this directly. Right click will give you the option to save the image, and in most browsers you can drag the image onto your desktop or another program.">

<img src="https://lh4.googleusercontent.com/QRIcd5-V2-hXwBDyB16OsEzeBsSoyiKP2yXbLGD5Ws2DVEkGyKi_0CapzMq-qxIwce8AtbdfsZzLpLd7iS3KLMYA7mPjPuZTkXNXKmKR6Z8SM71D4TBEGu11U8oqGj2J6r2VNbM" width="378px;" height="258px;" style="border: none; transform: rotate(0.00rad); -webkit-transform: rotate(0.00rad);">

그림 1. 위 함수의 그래프

simple_example.lua를 실행해봅니다. 초기 시작 점은 5입니다.
```bash
> th ./simple_example.lua 
3.679312	
```
지역 최솟값(local minimum)인 3.7 정도에서 수렴하였습니다.

초기 시작점을 2로 바꿔 다시 실행해 봅니다.
```bash
> th ./simple_example.lua 
0.001666
```
전역 최솟값(global minimum)인 0으로 수렴하였습니다.


## 과제 2: 최적화기 튜닝
매 미니배치(또는 에포크)마다 테스트 세트의 성능을 평가하도록 코드를 수정하십시오. 
훈련 손실과 시험 손실을 같은 그림에 출력하십시오. 이 일을 하는 데 필요한 코드를 
설명하는 주석을 간단히 다십시오.

잘 동작하는 최적화기 구성을 찾으십시오 (미니배치 크기 등). 훈련 및 시험 세트의 
분류 오차를 계산하십시오. 가장 잘 분류된 최적화기 구성을 보고하십시오. 그 때의 시험 
세트와 훈련 세트에 대한 분류 오차를 보고하십시오. 어떤 최적화기가 다른 것보다 더 
구성하기 쉬운지 보고하십시오. 당신이 선택한 모델이 왜 좋은지 간략히 설명하십시오. 
분류 에러를 출력하기 위한 코드를 보이십시오.

<img src="https://lh6.googleusercontent.com/tD2Lb9I1zatRKgEobI-wIs1yXal2ORzH0WaOZ3QEdgmt3Y9cbKCxn_gMtUTFkAauJUsXFeH7EMNm-_QUYs98-qC2PrpgWSutylb59fPnLuU3XrSsliqTVZkyByACF9tjt_-nQwip" width="519px;" height="379px;" style="border: none; transform: rotate(0.00rad); -webkit-transform: rotate(0.00rad);">

그림 1. SGD 최적화기를 위한 최적 구성 (Corr은 정답률, # of minibatches는 미니배치 
크기가 아니라 iteration을 뜻함).

<img src="https://lh3.googleusercontent.com/X5c5EXxx8cit5u4A-2A5q6QvVJy4xcgZUn4ZrIPNjt4jmqs3k1UI-MoJ-W-sxMkv26Uv5XVyROgkkOIj1J2EwwqX7gX1uNcAh9e88kbvCmC_CearjLoRB3XV8N2ciICiwiWhP3A-" width="512px;" height="374px;" style="border: none; transform: rotate(0.00rad); -webkit-transform: rotate(0.00rad);">

그림 2. adagrad 최적화기를 위한 최적 구성 (Corr은 정답률, # of minibatches는 미니배치 
크기가 아니라 iteration을 뜻함).


가장 구성하기 쉬운 최적화기는 adagrad 입니다. 이유는 파라미터가 학습률 하나뿐이기 
때문입니다. 소스 코드는 아래 링크를 참고해 주십시오.
<https://github.com/LeeTaewoo/torch_kor_manual/blob/master/oxford_ml_practicals/practical3/practical3_ans.lua#L160#L207/>

## 앞서나가기 과제
[유한 차분](https://en.wikipedia.org/wiki/Finite_difference) 근사치 계산법을 사용한 기울기 확인기 구현.

(<http://roboticist.tistory.com/584/>를 참고해 주십시오).

