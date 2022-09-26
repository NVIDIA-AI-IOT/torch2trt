import torch
from models.resnet import resnet18

model=resnet18(qat_mode=True)

rand_in = torch.randn([128,3,32,32],dtype=torch.float32).cuda()
model = model.cuda().train()

out = model(rand_in)
for k,v in model.named_parameters():
    print(k,v.size())
#print(model)
#model = torch.jit.script(model.eval())

