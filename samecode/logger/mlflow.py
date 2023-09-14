import mlflow 
import os
import pandas as pd

class Logger():
    def __init__(self, experiment='', run_name='', root='', run_id=''):
        '''
        Log parameters of model to MlFlow
        '''
        mlflow.end_run()
        self.experiment = experiment
        self.run_name = run_name
        self.root = root
        self.runs = {}
        
        mlflow.tracking.set_tracking_uri('file://{}'.format(root))
        mlflow.set_experiment(self.experiment)
        
        self.experiment = mlflow.get_experiment_by_name(self.experiment)
            
        if run_id: 
            mlflow.start_run(run_id=run_id)
        else:
            mlflow.start_run(run_name=self.run_name)
            
        self.parent_run = mlflow.active_run()
        self.parent_id = self.parent_run.info.run_id

        self.parent_artifacts = "{}/{}/{}/artifacts/".format(self.root, self.experiment.experiment_id, self.parent_id)


    def init(self, **kwargs):
        '''
        Create and log a new run
        '''

        # try:
        #     self.child.end_run()
        # except:
        #     pass

        run_id = kwargs.get('run_id', None)

        if run_id == None: 
            run_name = kwargs.get('run_name', None)
            mlflow.start_run(run_name=run_name, nested=True)
        else:
            mlflow.start_run(run_id=run_id, nested=True)

        child_run = mlflow.active_run()
        child_run_id = child_run.info.run_id
        
        # self.runs[run_name] = child_run_id

        mlflow.log_param('mlflow_id', child_run_id)
        child_artifacts = "{}/{}/{}/artifacts/".format(self.root, self.experiment.experiment_id, child_run_id)
        
        self.child = mlflow

        return child_run_id, child_artifacts
        
    def update(self, run_id, trainer):
        '''
        Update a run using run_id
        '''

        if type(trainer) == dict:
            trainer_params = list(trainer.items())

        else:
            trainer_params = [[k,v] for k,v in trainer.__dict__.items() if type(v) in [int, str, float, bool]]
            # mlflow.start_run(run_id=run_id, experiment_id=self.experiment.experiment_id, nested=True)
        
        for k, v in trainer_params:
            try:
                self.child.log_param(k, v)
            except:
                pass
        
    def close(self):
        '''
        Close mlflow and make a copy of the notebook and store it on the mlflow run
        '''
        self.child.end_run()
        mlflow.end_run()

    def add_note(self, run_id, comment):
        self.child.tracking.MlflowClient().set_tag(run_id, "mlflow.note.content", comment)

    def save(self, name, artifacts):
        os.system('cp {}/{}.ipynb {}'.format(os.getcwd(), name, artifacts))

    def get_metric(self, run_id, metric):
        spath = '{root}/{pid}/{rid}/metrics/{m}'.format(root=self.root, pid=self.experiment.experiment_id, rid=run_id, m=metric)
        metrics = []
        for i in open(spath):
            metrics.append([float(k) for k in i.strip().split()])

        return pd.DataFrame(metrics, columns=['timestamp', metric, 'step'])
        

