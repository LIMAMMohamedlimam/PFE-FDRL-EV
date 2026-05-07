from datetime import datetime
    
def ts() : 
    """
    generate timestamp
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")