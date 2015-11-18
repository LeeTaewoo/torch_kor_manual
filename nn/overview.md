<a name="nn.overview.dok"></a>
# 개요 #

한 네트워크의 각 모듈은 [Modules](module.md#nn.Modules)로 구성됩니다. 
그리고 `모듈`에는 사용 가능한 몇 가지 하위 클래스가 있습니다.
[Sequential](containers.md#nn.Sequential), [Parallel](containers.md#nn.Parallel), 
[Concat](containers.md#nn.Concat) 같은 컨테이너 클래스들은 
[Linear](simple.md#nn.Linear), [Mean](simple.md#nn.Mean), [Max](simple.md#nn.Max), 
[Reshape](simple.md#nn.Reshape) 같은 단순한 층 뿐 아니라 
[컨볼루셔널 층들](convolution.md)과 [Tanh](transfer.md#nn.Tanh) 같은 [전달 함수들](transfer.md)도 담을 수 있습니다.

손실 함수들은 [오차 판정 기준(Criterions)](criterion.md#nn.Criterions)의 하위 클래스들로 구현됩니다.
손실 함수들은 분류 과제에 대해 신경망을 훈련시키는 데 도움이 됩니다.
흔한 오차 판정 기준은 [MSECriterion](criterion.md#nn.MSECriterion)에 구현된 평균 제곱 오차 판정 기준과
[ClassNLLCriterion](criterion.md#nn.ClassNLLCriterion)에 구현된 크로스-엔트로피 오차 판정 기준입니다.

물론 간단한 for 루프로 [사용자 스스로 신경망을 훈련시키는 것](training.md#nn.DoItYourself)도 쉽습니다.
하지만, [StochasticGradient](training.md#nn.StochasticGradient) 클래스는 선택한 신경망을
훈련시킬 수 있는 높은 수준의 방식을 제공합니다.

## 더 자세한 개요 ##
이 절에서는 신경망 패키지를 조금 더 자세히 설명합니다. 우선 어디에나 나오는 [모듈](#nn.overview.module)부터 
다뤄집니다. 그리고 그 [모듈들을 결합](#nn.overview.plugandplay)하는 예제가 나옵니다.
마지막 부분에서는 [신경망을 훈련 시키기](#nn.overview.training) 위한 장치들(facilities)과 
[공유된 파라미터들](#nn.overview.sharedparams)로 네트워크를 훈련시키는 동안 주의할 점을 다룹니다.

<a name="nn.overview.module"></a>
### 모듈 ###

토치에서는 신경망 하나가 [모듈](module.md#nn.Module)이라 불립니다.
`모듈`은 네 가지 주요 메소드를 정의하는 추상 클래스입니다.

  * [forward(input)](module.md#nn.Module.forward)는 주어진 `input`  [텐서](https://github.com/torch/torch7/blob/master/doc/tensor.md)에서 그 모듈의 출력을 계산합니다.
  * [backward(input, gradOutput)](module.md#nn.Module.backward)는 그 모듈의 파라미터들과 입력에 대한 기울기를 계산합니다.
  * [zeroGradParameters()](module.md#nn.Module.zeroGradParameters)는 그 모듈의 파라미터들에 대한 기울기들을 0으로 만듭니다.
  * [updateParameters(learningRate)](module.md#nn.Module.updateParameters)는 `backward()`로 기울기가 계산된 다음, 파라미터들을 갱신합니다.

또한, `모듈`은 두 멤버(변수)를 선언합니다.

  * [output](module.md#nn.Module.output)은 `forward()`가 리턴하는 출력입니다.
  * [gradInput](module.md#nn.Module.gradInput)에는 `backward()`에서 계산된 그 모듈의 입력에 대한 기울기가 저장됩니다.

사용 빈도는 낮을 수도 있지만, 쓰기 편한 두 메소드도 있습니다.

  * [share(mlp,s1,s2,...,sn)](module.md#nn.Module.share)는 이 모듈이 `mlp` 모듈과 파라미터 s1,...,sn을 공유하게 만듭니다. 이 메소드는 우리가 같은 가중치를 공유하는 모듈들을 만들고 싶을 때 유용합니다.
  * [clone(...)](module.md#nn.Module.clone)은 이 모듈의 깊은 복사를 생성합니다 (단지 포인터가 아닌). 그 깊은 복사의 대상에는 (만약 있다면) 그 모듈의 파라미터들에 대한 현재 상태도 포함됩니다.

중요한 몇 가지 주목할 점:

  * `output`에는 [forward(input)](module.md#nn.Module.forward) 뒤에 오직 유효한 값들만 저장됩니다.
  * `gradInput`에는 [backward(input, gradOutput)](module.md#nn.Module.backward) 뒤에 오직 유효한 값들만 저장됩니다.
  * [backward(input, gradOutput)](module.md#nn.Module.backward)는 [forward(input)](module.md#nn.Module.forward) 동안 얻어진 특정 계산들을 사용합니다. `backward()`를 호출하기 전 _반드시_ _같은_ `input`에 대한 `forward()`가 호출되어야 합니다, 그렇지 않으면 계산된 기울기는 부정확할 것입니다!

<a name="nn.overview.plugandplay"></a>
### 플러그 앤 플레이 ###

우리는 층 하나를 만듦으로써 간단한 신경망 하나를 만들 수 있습니다.
한 개의 선형 신경망(퍼셉트론!)은 단지 이 한 줄로 만들어집니다.
```lua
mlp = nn.Linear(10,1) -- 10개 입력을 가진 퍼셉트론
```

더 복잡한 신경망은 [Sequential](containers.md#nn.Sequential)과 [Concat](containers.md#nn.Concat) 같은
컨테이너 클래스들을 사용하여 쉽게 만들어집니다. `Sequential`은 층들을 피드포워드 완전 연결 방식으로 연결합니다.
`Concat`은 한 층으로 여러 모듈들을 연관시킵니다. 그것들은 같은 입력을 받고, 그것들의 출력은 연관됩니다(concatenated).

그러므로, 숨겨진 층이 하나인 다층 퍼셉트론을 만드는 일은 다음과 같이 쉽습니다.
```lua
mlp = nn.Sequential()
mlp:add( nn.Linear(10, 25) ) -- 입력 10개, 숨겨진 유닛 25개
mlp:add( nn.Tanh() ) -- 쌍곡 탄젠트(hyperbolic tangent) 전달 함수
mlp:add( nn.Linear(25, 1) ) -- 출력 1개
```

물론, `Sequential`과 `Concat`은 다른 `Sequential` 또는 `Concat`도 포함할 수 있습니다.
이는 우리가 꿈꿔온 아주 아주 복잡한 신경망도 만들어볼 수 있게 합니다!
[[#nn.Modules|사용 가능한 모듈들의 완전한 목록]]을 보십시오.

<a name="nn.overview.training"></a>
### 신경망 훈련하기 ###

일단 신경망이 만들어지면, 우리는 그것을 훈련시키기 위한 
[오차 판정 기준(criterion)](criterion.md#nn.Criterions)을 선택해야 합니다.
오차 판정 기준은 훈련하는 동안 비용(cost)을 최소화하게 만드는 한 클래스입니다. 

그런 다음, 우리는 그 신경망을 [StochasticGradient](training.md#nn.StochasticGradient)
클래스를 사용하여 훈련시킬 수 있습니다.

```lua
 criterion = nn.MSECriterion() -- 평균 제곱 오차 판정 기준
 trainer = nn.StochasticGradient(mlp, criterion)
 trainer:train(dataset) -- 약간의 예제들을 사용하여 훈련
```

`StochasticGradient`는 `dataset`로 한 객체를 기대합니다. 
그 객체는 `dataset[index]` 연산자와 `dataset:size()`를 구현합니다.
그 `size()` 메소드는 예제의 개수를 리턴하고, 
`dataset[i]`는 i 번째 예제를 리턴해야만 합니다.

한 `example`는 `example[field]` 연산자를 구현하는 객체여야 합니다.
여기서 `field`는 값 `1`(입력 특징들) 또는 `2`(오차 판정 기준에 주어질 상응하는 레이블)를 입력받을 수도 있습니다.
(만약 당신이 [table layers](table.md#nn.TableLayers) 같은 특별한 종류의 기울기 모듈들을 사용하지 않는 이상) 그 입력은 보통 한 텐서입니다. 레이블 타입은 그 오차 판정 기준에 따라 달라집니다. 
예를 들어, [MSECriterion](criterion.md#nn.MSECriterion)은 한 텐서를 기대하지만,
[ClassNLLCriterion](criterion.md#nn.ClassNLLCriterion)은 한 정수(그 클래스)를 기대합니다.

그런 데이터세트는 루아 테이블들을 사용하여 쉽게 만들 수 있습니다.
그러나 그것은, 예를 들어, 요구되는 연산자/메소드들이 구현되는 한, 어떤 `C` 객체일 수도 있습니다.
[한 예제를 보십시오](containers.md#nn.DoItStochasticGradient).

`Lua`로 쓰인 `StochasticGradient`는 잘라붙이기(cut-and-paste)와 
우리의 필요에 따라 그것을 변형하기가 매우 쉽습니다 (만약 `StochasticGradient`의 제한들이 만족스럽지 않으면).

<a name="nn.overview.lowlevel"></a>
#### 저수준 훈련 ####

만약 우리가 `StochasticGradient`을 직접 프로그램하기 원하면,
우리는 근본적으로 네트워크에 대한 fowards와 backwards를 우리 스스로 제어할 필요가 있습니다.
예를 들어, 여기 기울기 단계을 구현하는 짧은 코드가 있습니다. 
여기서 `x`는 입력, `y`는 희망 출력(desired output), `mlp`는 한 네트워크, `criterion`은 오차 판정 기준, 
그리고 `learningrate`은 학습률을 나타냅니다.

```lua
function gradUpdate(mlp, x, y, criterion, learningRate) 
  local pred = mlp:forward(x)
  local err = criterion:forward(pred, y)
  local gradCriterion = criterion:backward(pred, y)
  mlp:zeroGradParameters()
  mlp:backward(x, gradCriterion)
  mlp:updateParameters(learningRate)
end
```
예를 들어, 만약 우리가 우리만의 오차 판정 기준을 사용하고 싶다면, 
우리는 단순히 `gradCriterion`를 우리가 선택한 오차 판정 기준의
기울기 벡터로 바꿀 수 있습니다.

<a name="nn.overview.sharedparams"></a>
### 파라미터 공유에 대한 노트 ###

`:share(...)`와 컨테이너 모듈들을 사용함으로써, 우리는 매우 복잡한 구조들도
쉽게 만들 수 있습니다. 그 네트워크가 적절하게 훈련됨을 보장하기 위해,
우리는 공유(sharing)가 적용되는 방법에 특별히 주의를 기울일 필요가 있습니다.
왜냐하면 공유가 적용되는 방법은 최적화 과정에 의존적일 수도 있기 때문입니다.

* 만약 우리가 우리의 네트워크의 모듈들에 대해 반복되는 최적화 알고리즘을 사용하고 있다면 (이를테면 `:updateParameters`를 호출함으로써), 오직 그 네트워크의 파라미터들만 공유되어야 합니다.
* 만약 우리가, 예를 들어 `optim` 패키지를 위한, `:getParameters`를 호출하여 얻은, 그 네트워크를 최적화하기 위한 납작해진(flattened) 파라미터 텐서를 사용하면, 우리는 그 파라미터들과 gradParameter들을 모두 공유할 필요가 있습니다.

여기, 첫 번재 경우를 위한 한 예제가 있습니다.

```lua
-- 우리의 최적화 절차는 모듈들대해 반복될 것입니다, 그래서 오직 파라미터들만 공유합니다.
mlp = nn.Sequential()
linear = nn.Linear(2,2)
linear_clone = linear:clone('weight','bias') -- 파라미터들의 공유를 복제합니다.
mlp:add(linear)
mlp:add(linear_clone)
function gradUpdate(mlp, x, y, criterion, learningRate) 
  local pred = mlp:forward(x)
  local err = criterion:forward(pred, y)
  local gradCriterion = criterion:backward(pred, y)
  mlp:zeroGradParameters()
  mlp:backward(x, gradCriterion)
  mlp:updateParameters(learningRate)
end
```

그리고 두 번째 경우를 위한 예제가 있습니다.

```lua
-- 우리의 최적화 절차는 모든 파라미터들을 한 번에 사용할 것입니다.
-- 왜냐하면, 그것이 납작해진 파라미터들과 gradParameters 텐서들를 요구하기 때문입니다.
-- 따라서, 우리는 파라미터들과 gradparameter들을 모두 공유할 필요가 있습니다.
mlp = nn.Sequential()
linear = nn.Linear(2,2)
-- need to share the parameters and the gradParameters as well
linear_clone = linear:clone('weight','bias','gradWeight','gradBias')
mlp:add(linear)
mlp:add(linear_clone)
params, gradParams = mlp:getParameters()
function gradUpdate(mlp, x, y, criterion, learningRate, params, gradParams)
  local pred = mlp:forward(x)
  local err = criterion:forward(pred, y)
  local gradCriterion = criterion:backward(pred, y)
  mlp:zeroGradParameters()
  mlp:backward(x, gradCriterion)
  -- 기울기들을 모든 파라미터들에 한 번에 더합니다
  params:add(-learningRate, gradParams)
end
```
