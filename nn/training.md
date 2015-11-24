<a name="nn.traningneuralnet.dok"></a>
# 신경망 훈련하기 #

신경망은 [간단한 `for` 루프](#nn.DoItYourself)로도 쉽게 훈련시킬 수 있습니다.
직접 구현하는 것은 그 내용을 자유롭게 바꿀 수 있다는 장점이 있습니다.
그러나 가끔 우리는 신경망을 빨리 훈련시키는 방법을 원할 지도 모릅니다.
당신을 위해 그 일을 하는 단순 클래스 [StochasticGradient](#nn.StochasticGradient)가 표준으로 제공됩니다.

<a name="nn.StochasticGradient.dok"></a>
## StochasticGradient (통계적 기울기) ##

`StochasticGradient`는 통계적 기울기 알고리즘을 사용하는 [neural networks](#nn.Module) 훈련을 위한 고수준 클래스입니다. 
이 클래스는 [직렬화](https://github.com/torch/torch7/blob/master/doc/serialization.md#serialization)할 수 있습니다.

<a name="nn.StochasticGradient"></a>
### StochasticGradient(module, criterion) ###

주어진 [Module](module.md#nn.Module)과 [Criterion](criterion.md#nn.Criterion)을 사용하여 `StochasticGradient` 클래스 하나를 만듭니다.
그 클래스는 당신이 초기화 뒤에 설정하기 원할지도 모르는 [몇 개의 파라미터들](#nn.StochasticGradientParameters)을 가집니다. 

<a name="nn.StochasticGradientTrain"></a>
### train(dataset) ###

오차 판정 기준(criterion)과 모듈은 [생성자](#nn.StochasticGradient)로 주어진 것을 사용합니다.
내부 [파라미터들](#nn.StochasticGradientParameters)을 사용하여, `dataset`에 대해 모듈을 훈련시킵니다.

`StochasticGradient`는 `dataset`으로 한 객체를 기대합니다. 
그 객체는 `dataset[index]` 연산자를 구현하고 `dataset:size()` 메소드를 구현합니다.
`size()` 메소드들은 예제들의 개수를 리턴하고 `dataset[i]`는 `i` 번째 예제를 리턴해야 합니다.

`예제` 하나는 한 객체여야 합니다. 그 객체는 연산자 `example[field]`를 구현합니다.
여기서 `field`는 값 `1`(입력 특징들) 또는 `2`(오차 판정 기준에 주어지는 상응하는 레이블)
을 받을 수도 있습니다.
([table layers](table.md#nn.TableLayers)처럼 특별한 종류의 기울기 모듈들을 사용하는 경우를 제외하면) 입력은 보통 텐서 하나입니다.
레이블 타입은 오차 판정 기준에 의존적입니다. 
예를 들어, [MSECriterion](criterion.md#nn.MSECriterion)은 텐서 하나를 기대하지만,
[ClassNLLCriterion](criterion.md#nn.ClassNLLCriterion)은 정수 숫자 (부류) 하나를 기대합니다.

그런 데이터세트는 루아 테이블을 사용하여 쉽게 만들어집니다. 
그러나 데이트세트는 요구되는 연산자/메소드가 구현되기만 하면, 어떤 `C` 객체일 수도 있습니다. 
[예제를 보십시오](#nn.DoItStochasticGradient).

<a name="nn.StochasticGradientParameters"></a>
### 파라미터들 ###

`StochasticGradient`는 [train()](#nn.StochasticGradientTrain)을 호출하는 데 영향을 끼치는 몇 가지 필드들을 가집니다.

  * `learningRate (학습률)`: 이것은 훈련시키는 동안 사용되는 학습률입니다. 파라미터들의 갱신은 `parameters = parameters - learningRate * parameters_gradient`일 것입니다. 기본값은 `0.01`입니다.
  * `learningRateDecay (학습률 쇠퇴)`: 학습률 쇠퇴(점점 줄어들어 사라짐). 만약 0이 아니면, 학습률(노트: 필드 학습률은 값을 바꾸지 않습니다)은 각 반복(그 데이터세트를 거쳐가는) 후에 다음 식으로 계산될 것입니다: `current_learning_rate =learningRate / (1 + iteration * learningRateDecay)`.
  * `maxIteration`: 최대 반복 횟수 (그 데이터세트를 거쳐가는). 기본값은 `25`입니다.
  * `shuffleIndices`: 예제들이 랜덤하게 샘플될 지 아닐지를 말하는 불리언 타입. 기본값은 `true`입니다. 만약 `false`이면, 예제들은 데이터세트에 있는 순서대로 읽힙니다.
  * `hookExample`: 후크(hook) 함수. 이 함수는 (만약 nil이 아니면) 훈련 동안 각 예제가 네트워크에 포워드되고 백워드된 다음에 호출됩니다. 이 함수는 파라미터들로 `(self, example)`를 받습니다. 기본값은 'nil'입니다. 
  * `hookIteration`: 후크(hook) 함수. 이 함수는 (만약 nil이 아니면) 훈련 동안 그 데이터세트를 완전히 한 번 거친 다음 호출됩니다. 이 함수는 파라미터들로 `(self, iteration)`을 받습니다. 기본값은 'nil'입니다. 

<a name="nn.DoItStochasticGradient"></a>
## StochasticGradient을 사용한 훈련 예 ##

여기서, 우리는 고전적인 XOR 문제에 대한 예제를 보입니다.

__데이터세트__

우리는 먼저 [StochasticGradient](#nn.StochasticGradientTrain)에 설명된 규약에 따라 데이터세트를 만들 필요가 있습니다. 
```lua
dataset={};
function dataset:size() return 100 end -- 예제 100개
for i=1,dataset:size() do 
  local input = torch.randn(2);     -- 2차원에서 정규 분포된 예제
  local output = torch.Tensor(1);
  if input[1]*input[2]>0 then       -- XOR 함수를 위한 레이블 계산
    output[1] = -1;
  else
    output[1] = 1
  end
  dataset[i] = {input, output}
end
```

__신경망__

우리는 숨겨진 층 하나를 가진 단순한 신경망 하나를 만듭니다.
```lua
require "nn"
mlp = nn.Sequential();  -- 다층 퍼셉트론 하나를 만듭니다
inputs = 2; outputs = 1; HUs = 20; -- 파라미터들
mlp:add(nn.Linear(inputs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs, outputs))
```

__훈련__

우리는 평균 제곱 오차 판정 기준으로 선택하고 그 데이터세트를 훈련시킵니다.
```lua
criterion = nn.MSECriterion()  
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01
trainer:train(dataset)
```

__네트워크 시험__

```lua
x = torch.Tensor(2)
x[1] =  0.5; x[2] =  0.5; print(mlp:forward(x))
x[1] =  0.5; x[2] = -0.5; print(mlp:forward(x))
x[1] = -0.5; x[2] =  0.5; print(mlp:forward(x))
x[1] = -0.5; x[2] = -0.5; print(mlp:forward(x))
```

다음과 같이 나와야 합니다:
```lua
> x = torch.Tensor(2)
> x[1] =  0.5; x[2] =  0.5; print(mlp:forward(x))

-0.3490
[torch.Tensor of dimension 1]

> x[1] =  0.5; x[2] = -0.5; print(mlp:forward(x))

 1.0561
[torch.Tensor of dimension 1]

> x[1] = -0.5; x[2] =  0.5; print(mlp:forward(x))

 0.8640
[torch.Tensor of dimension 1]

> x[1] = -0.5; x[2] = -0.5; print(mlp:forward(x))

-0.2941
[torch.Tensor of dimension 1]
```

<a name="nn.DoItYourself"></a>
## 신경망을 직접 짠 코드로 훈련시키는 예 ##

여기서, 우리는 고전적인 XOR 문제에 대한 예를 보입니다.

__신경망__

우리는 숨겨진 층 하나를 가진 단순한 신경망 하나를 만듭니다.
```lua
require "nn"
mlp = nn.Sequential();  -- 다층 퍼셉트론 하나를 만듭니다
inputs = 2; outputs = 1; HUs = 20; -- 파라미터들
mlp:add(nn.Linear(inputs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs, outputs))
```

__손실 함수__

평균 제곱 오차 판정 기준을 선택합니다.
```lua
criterion = nn.MSECriterion()  
```

__훈련__

우리는 데이터를 _즉석에서_ 만들고 그것을 신경망에 넣습니다.

```lua
for i = 1,2500 do
  -- 랜덤 샘플
  local input= torch.randn(2);    -- 2차원에서 정규 분포된 예제
  local output= torch.Tensor(1);
  if input[1]*input[2] > 0 then  -- XOR 함수를 위한 레이블 계산
    output[1] = -1
  else
    output[1] = 1
  end

  -- 그것을 신경망에 넣습니다. 그리고 오차 판정 기준
  criterion:forward(mlp:forward(input), output)

  -- 이 예제에 대해 3 단계로 훈련 시킵니다
  -- (1) 기울기들의 누적을 0으로 초기화합니다
  mlp:zeroGradParameters()
  -- (2) 기울기들을 누적합니다
  mlp:backward(input, criterion:backward(mlp.output, output))
  -- (3) 학습률 0.01로 파라미터들을 갱신합니다
  mlp:updateParameters(0.01)
end
```

__네트워크 시험__

```lua
x = torch.Tensor(2)
x[1] =  0.5; x[2] =  0.5; print(mlp:forward(x))
x[1] =  0.5; x[2] = -0.5; print(mlp:forward(x))
x[1] = -0.5; x[2] =  0.5; print(mlp:forward(x))
x[1] = -0.5; x[2] = -0.5; print(mlp:forward(x))
```

다음과 같이 나와야 합니다:
```lua
> x = torch.Tensor(2)
> x[1] =  0.5; x[2] =  0.5; print(mlp:forward(x))

-0.6140
[torch.Tensor of dimension 1]

> x[1] =  0.5; x[2] = -0.5; print(mlp:forward(x))

 0.8878
[torch.Tensor of dimension 1]

> x[1] = -0.5; x[2] =  0.5; print(mlp:forward(x))

 0.8548
[torch.Tensor of dimension 1]

> x[1] = -0.5; x[2] = -0.5; print(mlp:forward(x))

-0.5498
[torch.Tensor of dimension 1]
```
