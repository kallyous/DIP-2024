import numpy as np

from utils import implotmany


class Process:

    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def execute(self, *new_args, **new_kwargs):
        return self.function(*new_args, *self.args, **new_kwargs, **self.kwargs)



class Pipeline:

    src = []
    data = []
    preprocessors = []
    processors = []
    postprocessors = []

    
    def __init__(self, src=[]):

        self.src = []
        self.data = []
        self.preprocessors = []
        self.processors = []
        self.postprocessors = []

        if not isinstance(src, list):
            raise TypeError("'src' deve ser uma lista de imagens em ndarray.")
        
        if len(src) == 0:
            raise ValueError("Pipeline sem imagens para processar.")

        for img in src:
            if not isinstance(img, np.ndarray):
                raise TypeError(f"As imagens devem ser ndarray do NumPy, e não {type(img)} .")
        
        for nda in src:
            self.src.append(nda.copy())


    def add_preproc(self, function, use_src=False, *args, **kwargs):
        if use_src:
            self.preprocessors.append(Process(function, *args, src=self.src, **kwargs))
        else:
            self.preprocessors.append(Process(function, *args, **kwargs))

    
    def add_process(self, function, use_src=False, *args, **kwargs):
        if use_src:
            self.processors.append(Process(function, *args, src=self.src, **kwargs))
        else:
            self.processors.append(Process(function, *args, **kwargs))

    
    def add_postproc(self, function, use_src=False, *args, **kwargs):
        if use_src:
            self.postprocessors.append(Process(function, *args, src=self.src, **kwargs))
        else:
            self.postprocessors.append(Process(function, *args, **kwargs))
    
    
    def run(self):

        self.data = []
        for img in self.src:
            self.data.append(img.copy())
        
        if self.preprocessors:
            for proc in self.preprocessors:
                self.data = proc.execute(self.data)
    
        if self.processors:
            for proc in self.processors:
                self.data = proc.execute(self.data)

        if self.postprocessors:
            for proc in self.postprocessors:
                self.data = proc.execute(self.data)


    def plot_data(self):
        implotmany(self.data)
