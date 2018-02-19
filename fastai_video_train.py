if __name__ == '__main__':
	from fastai.conv_learner import *
	from videodataset import *
	from fastai.transforms import *
	from torch import nn
	import torch.nn.functional as F

	class ConvLayer(nn.Module):
		def __init__(self, ni, nf):
			super().__init__()
			self.conv = nn.Conv3d(ni, nf, kernel_size=3, stride=2, padding=1)

		def forward(self, x): 
			return F.relu(self.conv(x))

	class ConvNet2(nn.Module):
		def __init__(self, layers, c):
			super().__init__()
			self.layers = nn.ModuleList([ConvLayer(layers[i], layers[i + 1])
				for i in range(len(layers) - 1)])
			self.out = nn.Linear(layers[-1], c)

		def forward(self, x):
			for l in self.layers: x = l(x)
			x = F.adaptive_max_pool3d(x, 1)
			x = x.view(x.size(0), -1)
			return F.log_softmax(self.out(x), dim=-1)

	def get_tfms(stats, sz):
		tfm_norm = Normalize(*stats)
		tfm_denorm = Denormalize(*stats)
		tfm_crop = CenterCrop(sz)
		trn_tfms = image_gen(tfm_norm, tfm_denorm, sz, tfms=[tfm_crop])
		val_tfms = image_gen(tfm_norm, tfm_denorm, sz, tfms=[tfm_crop])
		return trn_tfms, val_tfms

	PATH = "./Images/"
	imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	sz = 224
	data = VideoClassifierData.from_paths(PATH, bs=1, trn_name="Train", val_name="Test", tfms=get_tfms(imagenet_stats, sz))
	learn = ConvLearner.from_model_data(ConvNet2([10, 20, 40, 80, 160], 10), data)
	learn.fit(0.01, 2)