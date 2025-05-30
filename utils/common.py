from PIL import Image
import matplotlib.pyplot as plt


# Log images
def log_input_image(x, opts):
	return tensor2im(x)


def tensor2im(var):
	# var shape: (3, H, W)
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))


def vis_faces(log_hooks):
	display_count = len(log_hooks)
	fig = plt.figure(figsize=(8, 4 * display_count))
	gs = fig.add_gridspec(display_count, 3)
	for i in range(display_count):
		hooks_dict = log_hooks[i]
		fig.add_subplot(gs[i, 0])
		if 'diff_input' in hooks_dict:
			vis_faces_with_id(hooks_dict, fig, gs, i)
		else:
			vis_faces_no_id(hooks_dict, fig, gs, i)
	plt.tight_layout()
	return fig


def vis_faces_masks(seg_loss,loss_dict2,log_hooks):
	display_count = len(log_hooks)
	fig = plt.figure(figsize=(16, 10 * display_count))
	gs = fig.add_gridspec(display_count, 6)
	for i in range(display_count):
		hooks_dict = log_hooks[i]
		fig.add_subplot(gs[i, 0])
		if 'diff_input' in hooks_dict:
			vis_faces_with_id_masks(seg_loss,loss_dict2, hooks_dict, fig, gs, i)
		else:
			vis_faces_no_id(hooks_dict, fig, gs, i)
	plt.tight_layout()
	return fig


def vis_faces_with_id(hooks_dict, fig, gs, i):
	plt.imshow(hooks_dict['input_face'])
	plt.title('Input\nOut Sim={:.2f}'.format(float(hooks_dict['diff_input'])))
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['target_face'])
	plt.title('Target\nIn={:.2f}, Out={:.2f}'.format(float(hooks_dict['diff_views']),
	                                                 float(hooks_dict['diff_target'])))
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['output_face'])
	plt.title('Output\n Target Sim={:.2f}'.format(float(hooks_dict['diff_target'])))
    
def vis_faces_with_id_masks(seg_loss, farl_metrics,hooks_dict, fig, gs, i):
	plt.imshow(hooks_dict['input_face'])
	plt.title('Input\nOut Sim={:.2f}'.format(float(hooks_dict['diff_input'])))
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['input_mask'])
	plt.title('BiSeNet Mask')
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['farl_init_mask'])
	plt.title('Farl Mask')
	fig.add_subplot(gs[i, 3])
	plt.imshow(hooks_dict['output_face'])
	plt.title('Output\n Target Sim={:.2f}'.format(float(hooks_dict['diff_target'])))
	fig.add_subplot(gs[i, 4])
	plt.imshow(hooks_dict['output_mask'])
	plt.title('BiSeNet\nIoU={:.2f}, Dice={:.2f}'.format(float(seg_loss[0][i]),
	                                                 float(seg_loss[1][i])))
	fig.add_subplot(gs[i, 5])
	plt.imshow(hooks_dict['farl_out_mask'])
	plt.title('Farl\nIoU={:.2f}, Dice={:.2f}'.format(float(farl_metrics[0][i]),
	                                                 float(farl_metrics[1][i])))


def vis_faces_no_id(hooks_dict, fig, gs, i):
	plt.imshow(hooks_dict['input_face'], cmap="gray")
	plt.title('Input')
	fig.add_subplot(gs[i, 1])
	plt.imshow(hooks_dict['target_face'])
	plt.title('Target')
	fig.add_subplot(gs[i, 2])
	plt.imshow(hooks_dict['output_face'])
	plt.title('Output')
