sys.path.insert(0, 'learn/')
from learn_model import LearnModel

class LanguageNetwork:
	def __init__(self, args, gamma):
        if self.args.lang_coeff > 0:
            self.setup_language_network()
            self.gamma = gamma

            # aggregates to compute Spearman correlation coefficients
            self.action_vectors_list = []
            self.rewards_list = []

    def reset(self):
        self.action_vector = np.zeros(N_ACTIONS)
        self.potentials_list = []



    def setup_language_network(self):
        self.lang_net_graph = tf.Graph()
        with self.lang_net_graph.as_default():
            self.lang_network = LearnModel('predict', None, self.args.model_dir)
        sentence_id = (self.args.expt_id-1) * 3 + self.args.descr_id
        lang_data = pickle.load(open('./data/test_data.pkl', 'rb'), encoding='bytes')
        self.lang = lang_data[sentence_id][self.args.lang_enc]

    def compute_language_reward(self, action):
        self.action_vector[action] += 1.
        if len(self.potentials_list) < 2:
            logits = None
        else:
            with self.lang_net_graph.as_default():
                logits = self.lang_network.predict([self.action_vector], [self.lang])[0]

        if logits is None:
            self.potentials_list.append(0.)
        else:
            e_x = np.exp(logits - np.max(logits))
            self.potentials_list.append(e_x[1] - e_x[0] + self.args.noise * np.random.normal())

        self.action_vectors_list.append(list(self.action_vector[k] for k in [0, 1, 2, 3, 4, 5, 11, 12]))
        self.rewards_list.append(self.potentials_list[-1])

        if len(self.potentials_list) > 1:
            lang_result = (self.gamma * self.potentials_list[-1] - self.potentials_list[-2])
            return lang_result
        else:
            return 0.

