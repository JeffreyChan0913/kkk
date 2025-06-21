import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cm import ScalarMappable
from matplotlib.dates import date2num, num2date, DateFormatter

def compute_bond_flow_score(yield_surface):
    """
        yield_surface: 2D numpy array [dates x maturities] of yields
        Returns: scalar flow_score between -1 and 1
    """
    
    # calculate gradient across the surface
    dy, dx = np.gradient(yield_surface)

    # calculate gradient magnitude
    grad_mag = np.sqrt(dx**2 + dy**2)

    # calculate mean resistance 
    avg_resistance = np.mean(grad_mag)

    # normalize using tanh into -1 to 1
    # flow_score = np.tanh( 2 * (0.5 - avg_resistance) ) # 2 is a parameter that can be tuned

    return - np.tanh(0.1 * avg_resistance)

def calculate_surface_flow_score(BOND, window_size = 252):
    """
        It calculates true SURFACE-based rolling bond flow score.

        Args:
            BOND: data frame, must have Date as first column, yield maturities as following columns.
            window_size: int, number of days to roll, 1 Year-> 252, 2 Year -> 504

        Returns:
            flow_scores: list of flow scores aligned with BOND dates
    """
    maturity_columns = BOND.columns[1:]
    n                = len(BOND)
    flow_scores      = [np.nan] * (window_size - 1)  # cant do anything with the first window

    for i in range(window_size - 1, n):
        window        = BOND.iloc[i - window_size + 1 : i + 1]
        yield_surface = window[maturity_columns].values
        score         = compute_bond_flow_score(yield_surface)
        flow_scores.append(score)

    return flow_scores


def compute_bond_flow_score_h(yield_surface, healthy_surface, alpha=2, across = 0):
    """
        yield_surface: 2D numpy array [dates x maturities] of yields
        healthy_surface: 2D numpy array of the same shape, representing the "normal" yield surface
        alpha: sensitivity parameter (higher = more sensitive)
        
        Returns: scalar flow_score between -1 and 1
    """
    # deviation = yield_surface - healthy_surface
    # mse = np.mean(deviation**2)  # Mean squared deviation
    # #flow_score = np.tanh(-alpha * mse)  # Flip sign so low MSE => +1
    # flow_score = -alpha * mse

    grad_y = np.gradient(yield_surface, axis = across)  # across maturity -> across = 1, else time -> across = 0
    grad_h = np.gradient(healthy_surface, axis = across)

    grad_deviation = grad_y - grad_h
    flow_score     = -alpha * np.mean(grad_deviation**2)

    return flow_score

def calculate_surface_flow_score_h(BOND, window_size=252, alpha=2, across=0):
    """
        Calculates SURFACE-based rolling bond flow score based on deviation from a healthy surface.

        Args:
            BOND: DataFrame, must have Date as first column, yield maturities as following columns.
            window_size: int, number of days to roll (e.g., 252 for 1 year)
            alpha: float, sensitivity parameter passed to score function

        Returns:
            flow_scores: list of flow scores aligned with BOND dates
    """
    healthy_surface = create_normal_yield_surface(maturities= BOND.columns[1::], n_days=window_size, max_yield=0.06)
    maturity_columns = BOND.columns[1:]
    n = len(BOND)
    flow_scores = [np.nan] * (window_size - 1)  # First few are NaN

    for i in range(window_size - 1, n):
        window        = BOND.iloc[i - window_size + 1: i + 1]
        yield_surface = window[maturity_columns].values
        score         = compute_bond_flow_score_h(yield_surface, healthy_surface.iloc[:, 1::], alpha, across)
        flow_scores.append(score)

    return flow_scores


def plot_yield_surface_with_drag(BOND, title = None, title_prefix="Yield Curve Surface with Drag", cmap='RdYlGn_r'):
    """
    Plots a 3D yield curve surface with aerodynamic-style drag pressure coloring.

    Args:
        BOND: pd.DataFrame with shape (days, maturities + 1)
              First column must be 'Date', remaining columns are maturities.
        title_prefix: str, prefix for the plot title.
        cmap: str, matplotlib colormap to use.
    """
    # Extract maturities and yield surface
    maturity_columns = BOND.columns[1::]  # Exclude 'Date' and FS columns
    yield_surface = BOND[maturity_columns].values  # shape: (n_days, n_maturities)

    # Compute gradient and flow magnitude ("drag pressure")
    dy, dx = np.gradient(yield_surface)
    drag_magnitude = np.sqrt(dx**2 + dy**2)

    # Convert dates to matplotlib date numbers
    dates = pd.to_datetime(BOND['Date']).values
    date_numbers = date2num(dates)

    # Create grid axes
    T, M = np.meshgrid(date_numbers, np.arange(len(maturity_columns)), indexing='ij')

    # Normalize drag for coloring
    norm_drag = drag_magnitude / np.max(drag_magnitude + 1e-8)

    # Plot 3D surface
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        T, M, yield_surface,
        facecolors=plt.cm.get_cmap(cmap)(norm_drag),
        rstride=1, cstride=1,
        linewidth=0, antialiased=False,
        shade=False
    )

    # Format x-axis with actual dates
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Set maturity labels
    ax.set_yticks(np.arange(len(maturity_columns)))
    ax.set_yticklabels(maturity_columns)

    if title == None:
        # Set dynamic title based on date range
        min_date = pd.to_datetime(BOND['Date']).min().strftime('%Y-%m-%d')
        max_date = pd.to_datetime(BOND['Date']).max().strftime('%Y-%m-%d')
        ax.set_title(f"{min_date} - {max_date} {title_prefix}")
    else:
        ax.set_title(title)

    # Remove the xlabel to avoid overlapping with xticks
    ax.set_xlabel("")
    ax.set_ylabel("Maturity")
    ax.set_zlabel("Yield (%)")

    # Add colorbar for drag magnitude
    mappable = ScalarMappable(cmap=cmap)
    mappable.set_array(norm_drag)
    fig.colorbar(mappable, ax=ax, shrink=0.5, label='Flow Drag Magnitude')

    plt.tight_layout()
    plt.show()

def create_normal_yield_surface(n_days = 500, maturities = None, min_yield = 0.01, max_yield = 0.05):
    """
    Create a synthetic 'normal' upward-sloping yield surface.
  
    Args:
        n_days: Number of business days to simulate.
        maturities: List of maturity labels (e.g. ['1M','3M',...]).
        min_yield: starting yield at shortest maturity (e.g. 0.01 for 1%).
        max_yield: ending yield at longest maturity (e.g. 0.05 for 5%).
      
    Returns:
        normal_bond_df: pd.DataFrame with Date + maturities columns.
    """
    if maturities is None:
        maturities = ['1 Mo', '3 Mo', '6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
    n_maturities = len(maturities)

    # 1) Base curve: linearly from min_yield to max_yield
    base_curve = np.linspace(min_yield, max_yield, n_maturities)  # shape (n_maturities,)

    # 2) Noise and trend
    np.random.seed(42)
    noise = np.random.normal(0, 0.001, size=(n_days, n_maturities))
    trend = np.linspace(0, 0.002, n_days)[:, None]  # shape (n_days,1)

    # 3) Build full surface by broadcasting base_curve across all days
    surface = base_curve[None, :] + trend + noise         # shape (n_days, n_maturities)

    # 4) Clip to realistic bounds [0%, 7%]
    surface = np.clip(surface, 0, 0.07)

    # 5) Turn into DataFrame
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq='B')
    normal_bond_df = pd.DataFrame(surface, columns=maturities, index=dates)
    normal_bond_df.index.name = 'Date'
    normal_bond_df.reset_index(inplace=True)

    return normal_bond_df
