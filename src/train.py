import numpy as np
import joblib
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import zipfile


# Theo92100 = Théo Niemann from the MVA
class ProjectAgent:
    def __init__(self):
        self.state_dim = 6 
        self.action_dim = 4 
        self.gamma = 0.90
        self.model=RandomForestRegressor()
        self.nb_fit_calls = 0

    def collect_samples(self,env, horizon, disable_tqdm=False, print_done_states=False,proba=0):
        s, _ = env.reset()
        S, A, R, S2, D = [], [], [], [], []

        for _ in tqdm(range(horizon), disable=disable_tqdm):
            use_random = (np.random.random() < proba)
            a = self.act(observation=s,use_random=use_random)
            s2, r, done, trunc, _ = env.step(a)
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(done)

            if done or trunc:
                s, _ = env.reset()
                if done and print_done_states:
                    print("Episode done!")
            else:
                s = s2

        S  = np.array(S)
        A  = np.array(A).reshape((-1,1))
        R  = np.array(R)
        S2 = np.array(S2)
        D  = np.array(D)

        return S, A, R, S2, D
    def train(self,env, max_iter, epochs ):
        horizon = 3000
        for epoch in range(epochs):
            S, A, R, S2, D = self.collect_samples(env, horizon = horizon, proba=0.1)
            for iteration in tqdm(range(max_iter)):
                Snew, Anew, Rnew, S2new, Dnew = self.collect_samples(env, horizon = 1000, proba=0.05 )
                
                S = np.vstack((S, Snew))
                A = np.vstack((A, Anew))   
                R = np.hstack((R, Rnew))     
                S2 = np.vstack((S2, S2new))
                D = np.hstack((D, Dnew))      
                SA = np.append(S, A, axis=1)
                nb_samples = S.shape[0]
                if iteration == 0:
                    target_values = R.copy()
                else:
                    Q_next = np.zeros((nb_samples, self.action_dim))
                    for action in range(self.action_dim):
                        A_next = np.full((nb_samples, 1), action)
                        S2A_next = np.append(S2, A_next, axis=1)
                        Q_next[:, action] = self.model.predict(S2A_next)
                    max_Q_next = np.max(Q_next, axis=1)
                    target_values = R + self.gamma * (1 - D) * max_Q_next
                
                self.model.fit(SA, target_values)
                self.nb_fit_calls += 1
                


    def act(self, observation, use_random=False):
        if use_random or (self.nb_fit_calls == 0):
            return np.random.randint(self.action_dim)
        else:
            Qsa = []
            s=observation
            for a in range(self.action_dim):
                sa = np.append(s, a).reshape(1, -1) 
                Qsa.append(self.model.predict(sa)[0])  
        return int(np.argmax(Qsa))
            

    def save(self,path='fqi_model.joblib'):
        if self.model is not None:
            joblib.dump(self.model, path)
            print(f"Model saved to {path}.")
        else:
            print("No model saved.")



    def load(self):
        zip_path = "fqi_model.joblib.zip"
        joblib_filename = "fqi_model.joblib"
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract(joblib_filename, path=".")
        self.model = joblib.load(joblib_filename)
        self.nb_fit_calls = 1

        print(f"Model loaded from {zip_path} (extracted {joblib_filename}).")



if __name__ == "__main__":
    env = TimeLimit(
        env=HIVPatient(domain_randomization=False),
        max_episode_steps=200
    )


    agent = ProjectAgent()


    agent.train(env,12,5)


    agent.save("fqi_model.joblib")
