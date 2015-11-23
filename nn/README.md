[![Build Status](https://travis-ci.org/torch/nn.svg?branch=master)](https://travis-ci.org/torch/nn)
<a name="nn.dok"></a>
# 신경망 패키지 #

원문: [https://github.com/torch/nn/blob/master/README.md]("https://github.com/torch/nn/blob/master/README.md")

이 패키지는 단순 또는 복잡한 신경망을 [토치](https://github.com/torch/torch7/blob/master/README.md)로 건설(build)하고 훈련하기 위한 쉽고 모듈적인 방법을 제공합니다.
  
  * 모듈은 신경망을 건설하는 데 사용되는 벽돌입니다. 각 모듈은 그 자신이 신경망입니다. 그러나 각 모듈은 다른 신경망들과 복잡한 신경망을 만들기 위해 컨테이너를 사용하여 결합될 수 있습니다: 
    * [모듈](module.md#nn.Module) : 모든 모듈들에 의해 상속되는 추상 클래스;
    * [컨테이너](containers.md#nn.Containers) : [Sequential](containers.md#nn.Sequential), [Parallel](containers.md#nn.Parallel), 그리고 [Concat](containers.md#nn.Concat) 같은 컨테이너 클래스들;
    * [전달 함수](transfer.md#nn.transfer.dok) : [Tanh](transfer.md#nn.Tanh)와 [Sigmoid](transfer.md#nn.Sigmoid) 같은 비선형 함수들;
    * [단순 층](simple.md#nn.simplelayers.dok) : [Linear](simple.md#nn.Linear), [Mean](simple.md#nn.Mean), [Max](simple.md#nn.Max), 그리고 [Reshape](simple.md#nn.Reshape) 같은; 
    * [테이블 층](table.md#nn.TableLayers) : [SplitTable](table.md#nn.SplitTable), [ConcatTable](table.md#nn.ConcatTable), 그리고 [JoinTable](table.md#nn.JoinTable) 같은 테이블 조작을 위한 층들;
    * [컨볼루션 층](convolution.md#nn.convlayers.dok) : [Temporal](convolution.md#nn.TemporalModules),  [Spatial](convolution.md#nn.SpatialModules), 그리고 [Volumetric](convolution.md#nn.VolumetricModules) 컨볼루션들; 
  * 오차 판정 기준(Criterion)은 주어진 입력과 타겟의 주어진 손실 함수에 따른 기울기를 계산합니다:
    * [Criterions](criterion.md#nn.Criterions) : 모든 오차 판정 기준들의 목록, 추상 클래스 [Criterion](criterion.md#nn.Criterion)을 포함하는;
    * [MSECriterion](criterion.md#nn.MSECriterion) : 회귀를 위해 사용되는 평균 제곱 오차(Mean Squared Error) 오차 판정 기준; 
    * [ClassNLLCriterion](criterion.md#nn.ClassNLLCriterion) : 분류를 위해 사용되는 네거티브 로그 우도(Negative Log Likelihood) 오차 판정 기준;
  * 추가 문서 :
    * 모듈, 컨테이너, 그리고 훈련을 포함하는 패키지의 핵심에 대한  [개요](overview.md#nn.overview.dok);
    * [훈련](training.md#nn.traningneuralnet.dok) : [StochasticGradient](training.md#nn.StochasticGradient)를 사용하여 어떻게 신경망을 훈련하는가;
    * [시험](testing.md) : 우리의 모듈을 어떻게 시험하는가.
    * [실험적인 모듈](https://github.com/clementfarabet/lua---nnx/blob/master/README.md) : 실험적인 모듈들과 오차 판정 기준들을 포함하는 패키지.

