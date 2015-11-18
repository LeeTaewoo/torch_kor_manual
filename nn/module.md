<a name="nn.Module"></a>
## 모듈 ##

`모듈`은 신경망을 훈련시키기 위한 기초적인 메소드들을 정의하는 추상 클래스입니다.
모듈은 [직렬화 할 수 있습니다](https://github.com/torch/torch7/blob/master/doc/serialization.md#serialization).

모듈은 [output](#output)과 [gradInput](#gradinput)의 두 상태 변수를 가집니다.

<a name="nn.Module.forward"></a>
### [output] forward(input) ###

한 `input` 객체를 받아서, 그에 상응하는 그 모듈의 `output`을 계산합니다.
일반적으로 `input`과 `output`은 [텐서](https://github.com/torch/torch7/blob/master/doc/tensor.md)입니다.
그러나 [table layers](table.md#nn.TableLayers) 같은 몇몇 특별한 하위 클래스들은 다른 입력을 기대할 수도 있습니다.
더 자세한 정보는 각 모듈의 설명서를 참조해 주십시오.

`forward()` 뒤에, [ouput](#output) 상태 변수는 반드시 새 값으로 갱신되었어야 합니다.

우리는 이 함수는 고치는 대신 [updateOutput(input)](#nn.Module.updateOutput) 
함수를 구현하길 권합니다. 추상 부모 클래스 [Module](#nn.Module)에 있는 포워드 모듈은 
`updateOutput(input)`를 호출합니다.


<a name="nn.Module.backward"></a>
### [gradInput] backward(input, gradOutput) ###

이 함수를 호출한 모듈에 대해, 주어진 `input`에 대한 _역전파 단계_를 수행합니다.
일반적으로 이 메소드는 _같은 input으로_ [forward(input)](#nn.Module.forward)가 이미 호출되었다고 가정합니다.
이것은 최적화를 위한 이유에서 필요합니다. 만약 우리가 이 규칙을 따르지 않으면,
`backward()`는 부정확한 기울기들을 계산할 것입니다.

일반적으로 `input`, `gradOutput`,  그리고 `gradInput`은 [텐서](https://github.com/torch/torch7/blob/master/doc/tensor.md)입니다.
그러나 [table layers](table.md#nn.TableLayers) 같은 몇몇 특별한 하위 클래스들은 다른 입력을 요구할 수도 있습니다.
더 자세한 정보는 각 모듈의 설명서를 참조해 주십시오.

_역전파 단계_의 특징은 주어진 `gradOutput`(그 모듈의 출력에 대한 기울기들)으로 `input`에 대해 두 종류의 기울기를 계산하는 것에 있습니다. 이 함수는 단지 두 함수 호출로 이 과제를 수행합니다.

  - [updateGradInput(input, gradOutput)](#nn.Module.updateGradInput) 함수 호출.
  - [accGradParameters(input,gradOutput,scale)](#nn.Module.accGradParameters) 함수 호출.

우리는 사용자가 만든(custom) 클래스들에서 이 함수 호출를 오버라이드하지 않길 권합니다. 
대신 [updateGradInput(input, gradOutput)](#nn.Module.updateGradInput)과
[accGradParameters(input, gradOutput,scale)](#nn.Module.accGradParameters) 함수를
오버라이드 하는 편이 더 좋습니다.

<a name="nn.Module.updateOutput"></a>
### updateOutput(input) ###

그 클래스의 현재 파라미터 집합과 입력을 사용하여 출력을 계산합니다.
이 함수는 [output](#output) 필드에 저장된 결과를 리턴합니다.


<a name="nn.Module.updateGradInput"></a>
### updateGradInput(input, gradOutput) ###

주어진 입력에 대한 그 모듈의 기울기들을 계산합니다.
이것은 `gradInput`으로 리턴됩니다. 또한, [gradInput](#gradinput)
상태 변수도 적절히 갱신됩니다.


<a name="nn.Module.accGradParameters"></a>
### accGradParameters(input, gradOutput, scale) ###

그 모듈의 파라미터에 대한 기울기를 계산합니다.
많은 모듈들은 이 단계를 수행하지 않습니다. 그 모듈들이 어떤 파라미터도 가지지 않기 때문입니다.
그 파라미터를 위한 상태 변수 이름은 모듈에 의존적입니다.
모듈은 몇몇 변수에 있는 파라미터들에 대한 기울기들을 _누적_하리라 기대됩니다.

`scale`은 누적되기 전에 gradParameters로 곱해지는 크기 조절 요소입니다.

[zeroGradParameters()](#nn.Module.zeroGradParameters)은 이 누적을 0으로 만듭니다.
그리고 [updateParameters()](#nn.Module.updateParameters)는 이 누적에 따라 
파라미터를 갱신합니다.


<a name="nn.Module.zeroGradParameters"></a>
### zeroGradParameters() ###

만약 그 모듈이 파라미터들을 가지면, 이 함수는 
[accGradParameters(input,gradOutput,scale)](#nn.Module.accGradParameters)호출을 
통해 누적된 이 파라미터들에 대한 기울기의 누적을 0으로 만들 것입니다.
만약 그 모듈이 파라미터들을 가지지 않으면, 이 함수는 아무것도 하지 않습니다.


<a name="nn.Module.updateParameters"></a>
### updateParameters(learningRate) ###

만약 그 모듈이 파라미터들을 가지면, 이 함수는 그 파라미터들을 갱신합니다.
그 갱신은 [backward()](#nn.Module.backward) 호출들을 통해 누적된 그 파라미터들에 대한 기울기들의 누적에 따라 수행됩니다. 

그 갱신은 기본적으로:
```lua
파라미터들 = 파라미터들 - 학습률 * 파라미터들에_대한_기울기들
```
만약 그 모듈이 파라미터들을 가지지 않으면, 아무 일도 하지 않습니다.

<a name="nn.Module.accUpdateGradParameters"></a>
### accUpdateGradParameters(input, gradOutput, learningRate) ###

이것은 두 함수를 한 번에 수행하는 편의 모듈입니다.
이 함수는 학습률 `learningRate`의 음수를 곱한 다음, 가중치들에 대한 기울기를 계산하고 누적합니다.
두 연산을 한 번에 수행하는 것은 성능 면에서 더 효율적이고, 어떤 상황에서는 이로울 수도 있습니다.

유념하십시오. 이 함수는 그 목적을 이루기 위해 간단한 트릭을 사용합니다.
이 함수는 사용자가 만든(custom) 모듈을 위해서는 유효하지 않을 수도 있습니다.

또한 주의하십시오. accGradParameters()와 대조적으로, 
이 함수는 기울기들을 (미래에 쓰기 위한 용도로) 계속 보유하지 않습니다.

```lua
function Module:accUpdateGradParameters(input, gradOutput, lr)
   local gradWeight = self.gradWeight
   local gradBias = self.gradBias
   self.gradWeight = self.weight
   self.gradBias = self.bias
   self:accGradParameters(input, gradOutput, -lr)
   self.gradWeight = gradWeight
   self.gradBias = gradBias
end
```

소스 코드에서 볼 수 있듯, 기울기들은 가중치들에 직접 누적됩니다.
이 가정은 비선형 연산을 계산하는 모듈을 위해서는 사실이 아닐 수도 있습니다.


<a name="nn.Module.share"></a>
### share(mlp,s1,s2,...,sn) ###

이 함수는 (만약 존재하면) 이름이 `s1`,...,`sn`인 그 모듈의 파라미터들을 수정하여 
주어진 `mlp` 모듈에 있는 이름이 같은 파라미터들과 공유되도록 (그 파라미터들을 가리키도록) 만듭니다.

파라미터들은 반드시 텐서들이어야 합니다. 이 함수는 보통 
같은 가중치들 또는 바이어스들을 공유하는 모듈들을 가지고 싶을 때 사용됩니다.

유념하십시오. 만약 [컨테이너](containers.md#nn.Containers) 모듈에서 호출되면,
이 함수는 거기에 포함된 모든 모듈들에서 같은 파라미터들을 공유합니다.

예:
```lua

-- mlp 하나를 만듭니다
mlp1=nn.Sequential(); 
mlp1:add(nn.Linear(100,10));

-- 두 번째 mlp 하나를 만듭니다
mlp2=nn.Sequential(); 
mlp2:add(nn.Linear(100,10)); 

-- 두 번째 mlp는 첫 번째 mlp와 바이어스를 공유합니다
mlp2:share(mlp1,'bias');

-- 첫 번째 mlp의 바이어스를 바꿉니다
mlp1:get(1).bias[1]=99;

-- 그리고 두 번째 mlp의 바이어스도 바뀌었는지 봅니다...
print(mlp2:get(1).bias[1])

```

<a name="nn.Module.clone"></a>
### clone(mlp,...) ###

그 모듈의 (단지 그것을 가리키는 포인터가 아닌) 깊은 복사를 만듭니다.
그 깊은 복사에는 (예를 들어, 만약 있다면 가중치, 바이어스 등의) 그 모듈의 파라미터들의 현재 상태도 포함됩니다.

만약 `clone(...)`에 인자들이 입력되면, 이 함수는 또한 
새 모듈을 만든 다음, 그 복제된 모듈에, 그 인자들로 [share(...)](#nn.Module.share)를 호출합니다.
따라서 그 복제된 모듈은 공유된 파라미터들을 가진 깊은 복사입니다.

예:
```lua
-- mlp 하나를 만듭니다
mlp1=nn.Sequential(); 
mlp1:add(nn.Linear(100,10));

-- 가중치들과 바이어스들을 공유하는 복제된 mlp 하나를 만듭니다.
mlp2=mlp1:clone('weight','bias');

-- 첫 번째 mlp의 바이어스를 바꿉니다
mlp1:get(1).bias[1]=99;

-- 그리고 두 번째 mlp의 바이어스도 바뀌었는지 봅니다...
print(mlp2:get(1).bias[1])

```

<a name="nn.Module.type"></a>
### type(type[, tensorCache]) ###

이 함수는 한 모듈의 모든 파라미터들을 주어진 `type`으로 바꿉니다.
그 `type`에는 [torch.Tensor](https://github.com/torch/torch7/blob/master/doc/tensor.md)를 위해 
정의된 타입 중 하나가 쓰일 수 있습니다.

만약 한 네트워크에서 여러 모듈 사이에 텐서들(또는 그것의 스토리지들)이 공유되면, 
이 공유는 `type`이 호출된 다음에도 유지됩니다.

여러 모듈들 그리고 (또는) 텐서들 사이에 공유를 유지하려면, `nn.utils.recursiveType`을 사용하십시오.

```lua
-- mlp 하나를 만듭니다
mlp1=nn.Sequential(); 
mlp1:add(nn.Linear(100,10));

-- 두 번째 mlp 하나를 만듭니다
mlp2=nn.Sequential(); 
mlp2:add(nn.Linear(100,10)); 

-- 두 번째 mlp는 첫 번째 mlp와 바이어스를 공유합니다
mlp2:share(mlp1,'bias');

-- mlp1과 mlp2는 float으로 변환되고, 바이어스를 공유할 것입니다.
-- 노트: (모듈들과 마찬가지로) 텐서들이 입력으로 들어갈 수 있습니다.
nn.utils.recursiveType({mlp1, mlp2}, 'torch.FloatTensor')
```

<a name="nn.Module.float"></a>
### float() ###

[module:type('torch.FloatTensor')](#nn.Module.type) 호출을 위한 편의 메소드

<a name="nn.Module.double"></a>
### double() ###

[module:type('torch.DoubleTensor')](#nn.Module.type) 호출을 위한 편의 메소드

<a name="nn.Module.cuda"></a>
### cuda() ###

[module:type('torch.CudaTensor')](#nn.Module.type) 호출을 위한 편의 메소드

<a name="nn.statevars.dok"></a>
### 상태 변수들 ###

이 상태 변수들은 한 `모듈`의 내부를 확인하고 싶을 때 유용한 객체들입니다.
그 객체 포인터는 보통 _결코_ 바뀌지 않습니다. 그러나, 그것의 
(만약 그것이 한 텐서이면 그것의 size를 포함하는)내용들은 바뀝니다.

일반적으로, 상태 변수들은 [텐서](https://github.com/torch/torch7/blob/master/doc/tensor.md)입니다.
그러나 [table layers](table.md#nn.TableLayers) 같은 몇몇 특별한 하위 클래스들은 다른 어떤 것을 포함합니다.
더 자세한 정보는 각 모듈의 설명서를 참조해 주십시오.

<a name="nn.Module.output"></a>
#### output ####

그 모듈의 출력을 담습니다. 마지막으로 호출된 [forward(input)](#nn.Module.forward) 호출로 계산됩니다.

<a name="nn.Module.gradInput"></a>
#### gradInput ####

그 모듈의 입력에 대한 기울기들을 담습니다. 
마지막으로 호출된 [updateGradInput(input, gradOutput)](#nn.Module.updateGradInput) 호출로 계산됩니다.


### 파라미터들에 대한 파라미터와 기울기들 ###

몇몇 모듈들은 (우리가 실제로 훈련시키기 원하는!) 파라미터들을 포함합니다.
이 파라미터들의 이름과 이 파라미터들에 대한 기울기들은 모듈에 따라 다릅니다.


<a name="nn.Module.parameters"></a>
### [{weights}, {gradWeights}] parameters() ###

이 함수는 테이블 두 개를 리턴해야 합니다. 
하나는 학습할 수 있는 파라미터들인 `{weights}`를 위한 것이고 
다른 하나는 그 학습할 수 있는 파라미터들 `{gradWeights}`에 대한 에너지의 기울기를 위한 것입니다.

만약 커스텀 모듈들이 텐서들에 저장된 학습할 수 있는 파라미터들을 사용하면, 
커스텀 모듈들은 이 함수를 오버라이드 해야 합니다. 


<a name="nn.Module.getParameters"></a>
### [flatParameters, flatGradParameters] getParameters() ###

이 함수는 텐서 두 개를 리턴합니다. 하나는 납작해진 학습할 수 있는 파라미터들 `flatParameters`을 위한 것이고
다른 하나는 학습할 수 있는 파라미터들 `flatGradParameters`에 대한 에너지의 기울기들을 위한 것입니다.

커스텀 모듈들은 이 함수를 오버라이드하지 않아야 합니다. 
대신 커스텀 모듈들은 현재 함수에 의해 차례로 호출되는 [parameters(...)](#nn.Module.parameters)를 오버라이드 해야 합니다.

이 함수는 모든 가중치들과 gradWeights를 점검합니다, 그리고 그것들을 한 단일 텐서 뷰로 만듭니다 (하나는 가중치들을 위해 그리고 하나는 gradWeights를 위해). 매 가중치와 gradWeight의 스토리지가 바뀌므로, 이 함수는 주어진 네트워크에서 오직 한 번만 호출되어야 합니다.


<a name="nn.Module.training"></a>
### training() ###
이 함수는 그 모듈(또는 하위 모듈들)의 모드를 `train=true`로 설정합니다. 
이것은 [Dropout](simple.md#nn.Dropout)처럼 훈련과 평가에서 다르게 동작하는 모듈들에 유용합니다.


<a name="nn.Module.evaluate"></a>
### evaluate() ###
이 함수는 그 모듈(또는 하위 모듈들)의 모드를 `train=false`로 설정합니다.
이것은 [Dropout](simple.md#nn.Dropout)처럼 훈련과 평가에서 다르게 동작하는 모듈들에 유용합니다.


<a name="nn.Module.findModules"></a>
### findModules(typename) ###
어떤 한 `typename`의 네트워크에 있는 모듈들의 모든 인스턴스들을 찾습니다. 
이 함수는 매치된 노드들의 납작해진 목록과 
각 매칭 노드를 위한 컨테이너 모듈들의 납잡해진 모듈들을 리턴합니다.

부모 컨테이너를 가지지 않은 모듈들(이를테면 nn.Sequential의 최상위)은 그 컨테이너로 `self`를 리턴할 것입니다.

이 함수는 복잡하게 중첩된 네트워크를 다루는 데 매우 도움이 됩니다. 예를 들어, 한 교훈적인 예는 다음과 같습니다.
만약 당신이 모든 `nn.SpatialConvolution`의 output size를 출력하기 원하면,

```lua
-- 다해상도(multi-resolution) 컨볼루션 네트워크 하나를 만듭니다 (해상도 두 개를 가진)
model = nn.ParallelTable()
conv_bank1 = nn.Sequential()
conv_bank1:add(nn.SpatialConvolution(3,16,5,5))
conv_bank1:add(nn.Threshold())
model:add(conv_bank1)
conv_bank2 = nn.Sequential()
conv_bank2:add(nn.SpatialConvolution(3,16,5,5))
conv_bank2:add(nn.Threshold())
model:add(conv_bank2)
-- 다해상도 샘플을 앞쪽으로(forward) 전파합니다.
input = {torch.rand(3,128,128), torch.rand(3,64,64)}
model:forward(input)
-- Threshold 출력들의 size를 출력합니다.
conv_nodes = model:findModules('nn.SpatialConvolution')
for i = 1, #conv_nodes do
  print(conv_nodes[i].output:size())
end
```

또다른 사용은 한 특정 `typename`을 가진 모든 노드들을 다른 것으로 바꾸는 것입니다. 예를 들어, 
만약 우리가 위 모델에서 모든 `nn.Threshold`를 `nn.Tanh`로 바꾸고 싶다면,

```lua
threshold_nodes, container_nodes = model:findModules('nn.Threshold')
for i = 1, #threshold_nodes do
  -- 현재 threshold 노드를 위한 컨테이너 탐색
  for j = 1, #(container_nodes[i].modules) do
    if container_nodes[i].modules[j] == threshold_nodes[i] then
      -- 새 인스턴스로 교체
      container_nodes[i].modules[j] = nn.Tanh()
    end
  end
end
```

<a name="nn.Module.listModules"></a>
### listModules() ###

한 네트워크에 있는 모든 모듈 인스턴스들을 열거합니다.
모듈들의 납작해진 목록 하나를 리턴합니다.
그 목록에는 컨테이너 모듈(가장 먼저 열거되는), self, 그리고 어떤 다른 요소 모듈들이 포함됩니다.

예를 들어, 아래 코드는 
```lua
mlp = nn.Sequential()
mlp:add(nn.Linear(10,20))
mlp:add(nn.Tanh())
mlp2 = nn.Parallel()
mlp2:add(mlp)
mlp2:add(nn.ReLU())
for i,module in ipairs(mlp2:listModules()) do
   print(module)
end
```

다음과 같은 출력을 만듭니다.

```lua
nn.Parallel {
  input
    |`-> (1): nn.Sequential {
    |      [input -> (1) -> (2) -> output]
    |      (1): nn.Linear(10 -> 20)
    |      (2): nn.Tanh
    |    }
    |`-> (2): nn.ReLU
     ... -> output
}
nn.Sequential {
  [input -> (1) -> (2) -> output]
  (1): nn.Linear(10 -> 20)
  (2): nn.Tanh
}
nn.Linear(10 -> 20)
nn.Tanh
nn.ReLU
```
