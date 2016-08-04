def set_up_dir(path):
    import os
    try: 
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
            
       