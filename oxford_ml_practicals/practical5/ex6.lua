require 'nngraph'

-- 문제 노드가 강조된 그 그래프의 SVG 생성합니다.  
-- 그리고 svg에서 노드들에 마우스 포인터를 올려놓으면, filename:line_number 정보를 볼 수 있습니다.
-- 디버그 모드가 활성화 되지 않더라도, 노드들은 지역 변수 이름으로 주석이 달릴 것입니다.  
nngraph.setDebug(true)

local function get_net(from, to)
    local from = from or 10
    local to = to or 10
    local input_x = nn.Identity()()
    local linear_module = nn.Linear(from, to)(input_x)

    -- 노드들에 지역 변수 이름의 주석을 답니다.
    nngraph.annotateNodes()
    return nn.gModule({input_x},{linear_module})
end

local net = get_net(10,10)

-- 만약 당신이 그 네트워크에 이름을 주면, 에러가 생긴 경우 svg에 그 이름을 사용합니다.
-- 만약 당신이 그 네트워크에 이름을 주지 않으면, 이름은 그 그래프의 입력과 출력 개수로 임의로 만들어 사용됩니다.
net.name = 'my_bad_linear_net'

-- 에러가 생기도록 일부러 틀린 차원을 가진 입력 하나를 준비합니다.
local input = torch.rand(11)
pcall(function() net:updateOutput(input) end)
-- 이것은 에러를 만들고 그래프 하나를 출력해야 합니다.

