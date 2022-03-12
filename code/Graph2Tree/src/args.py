import argparse


def build_parser():
	# Data loading parameters
	parser = argparse.ArgumentParser(description='Run Single sequence model')

	parser.add_argument('-mode', type=str, default='train', choices=['train', 'test'], help='Modes: train, test')

	# Run Config
	parser.add_argument('-run_name', type=str, default='debug', help='run name for logs')
	parser.add_argument('-dataset', type=str, default='mawps_fold0', help='Dataset')
	parser.add_argument('-outputs', dest='outputs', action='store_true', help='Show full validation outputs')
	parser.add_argument('-no-outputs', dest='outputs', action='store_false', help='Do not show full validation outputs')
	parser.set_defaults(outputs=True)
	parser.add_argument('-results', dest='results', action='store_true', help='Store results')
	parser.add_argument('-no-results', dest='results', action='store_false', help='Do not store results')
	parser.set_defaults(results=True)

	# Meta Attributes
	parser.add_argument('-trim_threshold', type=int, default=1,
						help='Remove words with frequency less than this from vocab')

	# Device Configuration
	parser.add_argument('-gpu', type=int, default=2, help='Specify the gpu to use')
	parser.add_argument('-seed', type=int, default=6174, help='Default seed to set')
	parser.add_argument('-logging', type=int, default=1, help='Set to 0 if you do not require logging')
	parser.add_argument('-ckpt', type=str, default='model', help='Checkpoint file name')
	parser.add_argument('-save_model', dest='save_model', action='store_true', help='To save the model')
	parser.add_argument('-no-save_model', dest='save_model', action='store_false', help='Dont save the model')
	parser.set_defaults(save_model=True)

	# Model parameters
	parser.add_argument('-embedding', type=str, default='bert', choices=['bert', 'roberta'], help='Embeddings')
	parser.add_argument('-emb_name', type=str, default='bert-base-uncased',
						choices=['bert-base-uncased', 'roberta-base'], help='Which pre-trained model')
	parser.add_argument('-embedding_size', type=int, default=768, help='Embedding dimensions of inputs')
	parser.add_argument('-emb_lr', type=float, default=1e-5, help='Learning rate to train embeddings')
	parser.add_argument('-freeze_emb', dest='freeze_emb', action='store_true', help='Freeze embedding weights')
	parser.add_argument('-no-freeze_emb', dest='freeze_emb', action='store_false', help='Train embedding weights')
	parser.set_defaults(freeze_emb=False)

	parser.add_argument('-cell_type', type=str, default='lstm', help='RNN cell for encoder and decoder, default: lstm')
	parser.add_argument('-hidden_size', type=int, default=384, help='Number of hidden units in each layer')
	parser.add_argument('-depth', type=int, default=2, help='Number of layers in each encoder')
	parser.add_argument('-lr', type=float, default=1e-3, help='Learning rate')
	parser.add_argument('-batch_size', type=int, default=4, help='Batch size')
	parser.add_argument('-weight_decay', type=float, default=1e-5, help='Weight Decay')
	parser.add_argument('-beam_size', type=float, default=5, help='Beam Size')
	parser.add_argument('-epochs', type=int, default=70, help='Maximum # of training epochs')
	parser.add_argument('-dropout', type=float, default=0.5,
						help='Dropout probability for input/output/state units (0.0: no dropout)')

	parser.add_argument('-nums_disp', dest='nums_disp', action='store_true',
						help='Display number of numbers information in validation outputs')
	parser.add_argument('-no-nums_disp', dest='nums_disp', action='store_false',
						help='Don\'t display number of numbers information')
	parser.set_defaults(nums_disp=True)
	parser.add_argument('-challenge_disp', dest='challenge_disp', action='store_true',
						help='Display information in validation outputs')
	parser.add_argument('-no-challenge_disp', dest='challenge_disp', action='store_false',
						help='Don\'t display information')
	parser.set_defaults(challenge_disp=False)

	parser.add_argument('-show_train_acc', dest='show_train_acc', action='store_true',
						help='Calculate the train accuracy')
	parser.add_argument('-no-show_train_acc', dest='show_train_acc', action='store_false',
						help='Don\'t calculate the train accuracy')
	parser.set_defaults(show_train_acc=False)

	parser.add_argument('-full_cv', dest='full_cv', action='store_true', help='5-fold CV')
	parser.add_argument('-no-full_cv', dest='full_cv', action='store_false', help='No 5-fold CV')
	parser.set_defaults(full_cv=False)

	parser.add_argument('-len_generate_nums', type=int, default=0, help='store length of generate_nums')
	parser.add_argument('-copy_nums', type=int, default=0, help='store copy_nums')

	return parser
