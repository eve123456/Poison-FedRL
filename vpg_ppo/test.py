a= {}
b = [{} for i in range(3)]

res = {}
for k in b[0].keys():
    res[k] = torch.average([i[k] for i in b])
