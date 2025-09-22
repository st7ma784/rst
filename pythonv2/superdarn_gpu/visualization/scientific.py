"""
Scientific Data Visualization for SuperDARN
===========================================

Publication-quality scientific visualizations for SuperDARN radar data.
Optimized for GPU data with CuPy integration.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
import warnings

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np  # Fallback to numpy
    GPU_AVAILABLE = False

from ..core.backends import get_backend


def plot_range_time(data, parameter='velocity', beam=None, ax=None, 
                   title=None, colorbar=True, **kwargs):
    """
    Create range-time plot of SuperDARN data
    
    Parameters:
    -----------
    data : dict or RadarData
        SuperDARN data containing time, range, and parameter arrays
    parameter : str
        Parameter to plot ('velocity', 'power', 'width', 'elevation')
    beam : int, optional
        Specific beam to plot (if None, plots all beams)
    ax : matplotlib.Axes, optional
        Axes to plot on (if None, creates new figure)
    title : str, optional
        Plot title
    colorbar : bool
        Whether to add colorbar
    **kwargs : dict
        Additional arguments for imshow
    
    Returns:
    --------
    im : matplotlib.image.AxesImage
        The plotted image
    """
    xp = get_backend()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract data
    if isinstance(data, dict):
        if parameter not in data:
            raise ValueError(f"Parameter '{parameter}' not found in data")
        plot_data = data[parameter]
        times = data.get('time', np.arange(plot_data.shape[0]))
        ranges = data.get('range', np.arange(plot_data.shape[-1]))
    else:
        # Assume it's a RadarData object
        plot_data = getattr(data, parameter)
        times = data.time
        ranges = data.range
    
    # Handle GPU data
    if hasattr(plot_data, 'get'):
        plot_data = plot_data.get()  # Transfer from GPU
    if hasattr(times, 'get'):
        times = times.get()
    if hasattr(ranges, 'get'):
        ranges = ranges.get()
    
    # Select specific beam if requested
    if beam is not None and plot_data.ndim > 2:
        if beam < plot_data.shape[1]:
            plot_data = plot_data[:, beam, :]
        else:
            raise ValueError(f"Beam {beam} not available (max: {plot_data.shape[1]-1})")
    elif plot_data.ndim > 2:
        # Average over all beams
        plot_data = np.nanmean(plot_data, axis=1)
    
    # Set up color scheme based on parameter
    if parameter == 'velocity':
        cmap = 'RdBu_r'
        vmin, vmax = kwargs.get('vmin', -1000), kwargs.get('vmax', 1000)
        label = 'Doppler Velocity (m/s)'
    elif parameter == 'power':
        cmap = 'viridis'
        vmin, vmax = kwargs.get('vmin', 0), kwargs.get('vmax', 50)
        label = 'Backscatter Power (dB)'
    elif parameter == 'width':
        cmap = 'plasma'
        vmin, vmax = kwargs.get('vmin', 0), kwargs.get('vmax', 500)
        label = 'Spectral Width (m/s)'
    elif parameter == 'elevation':
        cmap = 'coolwarm'
        vmin, vmax = kwargs.get('vmin', 0), kwargs.get('vmax', 60)
        label = 'Elevation Angle (degrees)'
    else:
        cmap = 'viridis'
        vmin, vmax = kwargs.get('vmin', None), kwargs.get('vmax', None)
        label = parameter
    
    # Create the plot
    im = ax.imshow(plot_data.T, aspect='auto', origin='lower',
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[times[0], times[-1], ranges[0], ranges[-1]],
                   **{k: v for k, v in kwargs.items() if k not in ['vmin', 'vmax']})
    
    # Format axes
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Range Gate')
    
    if title is None:
        beam_str = f' (Beam {beam})' if beam is not None else ''
        title = f'{parameter.title()}{beam_str}'
    ax.set_title(title)
    
    # Add colorbar
    if colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(label)
    
    # Format time axis if times are datetime objects
    if len(times) > 0 and hasattr(times[0], 'hour'):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    return im


def plot_fan(data, parameter='velocity', time_idx=0, ax=None, 
            title=None, colorbar=True, **kwargs):
    """
    Create fan plot showing all beams at a specific time
    
    Parameters:
    -----------
    data : dict or RadarData
        SuperDARN data
    parameter : str
        Parameter to plot
    time_idx : int
        Time index to plot
    ax : matplotlib.Axes, optional
        Axes to plot on
    title : str, optional
        Plot title
    colorbar : bool
        Whether to add colorbar
    **kwargs : dict
        Additional plotting arguments
    
    Returns:
    --------
    im : matplotlib.image.AxesImage
        The plotted image
    """
    xp = get_backend()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract data
    if isinstance(data, dict):
        plot_data = data[parameter]
        beams = data.get('beam', np.arange(plot_data.shape[1]))
        ranges = data.get('range', np.arange(plot_data.shape[-1]))
    else:
        plot_data = getattr(data, parameter)
        beams = data.beam
        ranges = data.range
    
    # Handle GPU data
    if hasattr(plot_data, 'get'):
        plot_data = plot_data.get()
    
    # Extract data for specific time
    if plot_data.ndim == 3:
        plot_data = plot_data[time_idx, :, :]
    
    # Set up color scheme (same as range_time)
    if parameter == 'velocity':
        cmap = 'RdBu_r'
        vmin, vmax = kwargs.get('vmin', -1000), kwargs.get('vmax', 1000)
        label = 'Doppler Velocity (m/s)'
    elif parameter == 'power':
        cmap = 'viridis'
        vmin, vmax = kwargs.get('vmin', 0), kwargs.get('vmax', 50)
        label = 'Backscatter Power (dB)'
    else:
        cmap = 'viridis'
        vmin, vmax = kwargs.get('vmin', None), kwargs.get('vmax', None)
        label = parameter
    
    # Create the plot
    im = ax.imshow(plot_data, aspect='auto', origin='lower',
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[beams[0], beams[-1], ranges[0], ranges[-1]])
    
    ax.set_xlabel('Beam Number')
    ax.set_ylabel('Range Gate')
    
    if title is None:
        title = f'{parameter.title()} Fan Plot (t={time_idx})'
    ax.set_title(title)
    
    if colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(label)
    
    return im


def plot_acf(acf_data, lag_idx=0, ax=None, plot_type='magnitude', **kwargs):
    """
    Plot Auto-Correlation Function data
    
    Parameters:
    -----------
    acf_data : array-like
        Complex ACF data [time, beam, range, lag]
    lag_idx : int
        Lag index to plot (0 is zero-lag)
    ax : matplotlib.Axes, optional
        Axes to plot on
    plot_type : str
        'magnitude', 'phase', 'real', 'imaginary', or 'both'
    **kwargs : dict
        Additional plotting arguments
    
    Returns:
    --------
    plot objects
    """
    if ax is None:
        if plot_type == 'both':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        else:
            fig, ax = plt.subplots(figsize=(10, 8))
    
    # Handle GPU data
    if hasattr(acf_data, 'get'):
        acf_data = acf_data.get()
    
    # Extract specific lag
    if acf_data.ndim == 4:
        data = acf_data[:, :, :, lag_idx]
    else:
        data = acf_data
    
    if plot_type == 'magnitude' or plot_type == 'both':
        magnitude = np.abs(data)
        if plot_type == 'both':
            im1 = ax1.imshow(np.mean(magnitude, axis=1).T, aspect='auto', 
                           origin='lower', cmap='viridis')
            ax1.set_title(f'ACF Magnitude (Lag {lag_idx})')
            ax1.set_xlabel('Time Index')
            ax1.set_ylabel('Range Gate')
            plt.colorbar(im1, ax=ax1)
        else:
            im = ax.imshow(np.mean(magnitude, axis=1).T, aspect='auto',
                         origin='lower', cmap='viridis')
            ax.set_title(f'ACF Magnitude (Lag {lag_idx})')
            plt.colorbar(im, ax=ax)
            return im
    
    if plot_type == 'phase' or plot_type == 'both':
        phase = np.angle(data)
        if plot_type == 'both':
            im2 = ax2.imshow(np.mean(phase, axis=1).T, aspect='auto',
                           origin='lower', cmap='hsv', vmin=-np.pi, vmax=np.pi)
            ax2.set_title(f'ACF Phase (Lag {lag_idx})')
            ax2.set_xlabel('Time Index')
            ax2.set_ylabel('Range Gate')
            plt.colorbar(im2, ax=ax2)
            return im1, im2
        else:
            im = ax.imshow(np.mean(phase, axis=1).T, aspect='auto',
                         origin='lower', cmap='hsv', vmin=-np.pi, vmax=np.pi)
            ax.set_title(f'ACF Phase (Lag {lag_idx})')
            plt.colorbar(im, ax=ax)
            return im
    
    if plot_type == 'real':
        real_part = np.real(data)
        im = ax.imshow(np.mean(real_part, axis=1).T, aspect='auto',
                     origin='lower', cmap='RdBu_r')
        ax.set_title(f'ACF Real Part (Lag {lag_idx})')
        plt.colorbar(im, ax=ax)
        return im
    
    if plot_type == 'imaginary':
        imag_part = np.imag(data)
        im = ax.imshow(np.mean(imag_part, axis=1).T, aspect='auto',
                     origin='lower', cmap='RdBu_r')
        ax.set_title(f'ACF Imaginary Part (Lag {lag_idx})')
        plt.colorbar(im, ax=ax)
        return im


def plot_spectrum(data, range_idx=None, beam_idx=None, ax=None, **kwargs):
    """
    Plot power spectrum of ACF data
    
    Parameters:
    -----------
    data : array-like
        ACF data to analyze
    range_idx : int, optional
        Specific range gate to analyze
    beam_idx : int, optional  
        Specific beam to analyze
    ax : matplotlib.Axes, optional
        Axes to plot on
    **kwargs : dict
        Additional plotting arguments
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Handle GPU data
    if hasattr(data, 'get'):
        data = data.get()
    
    # Select specific range/beam if requested
    if data.ndim == 4:  # [time, beam, range, lag]
        if range_idx is not None and beam_idx is not None:
            acf_profile = data[:, beam_idx, range_idx, :]
        elif range_idx is not None:
            acf_profile = np.mean(data[:, :, range_idx, :], axis=1)
        elif beam_idx is not None:
            acf_profile = np.mean(data[:, beam_idx, :, :], axis=1)
        else:
            acf_profile = np.mean(np.mean(data, axis=1), axis=1)
    else:
        acf_profile = data
    
    # Calculate power spectrum for each time
    spectra = []
    for t in range(acf_profile.shape[0]):
        spectrum = np.fft.fftshift(np.fft.fft(acf_profile[t, :]))
        spectra.append(np.abs(spectrum)**2)
    
    spectra = np.array(spectra)
    
    # Create frequency axis
    n_lags = acf_profile.shape[1]
    freqs = np.fft.fftshift(np.fft.fftfreq(n_lags))
    
    # Plot spectrogram
    im = ax.imshow(spectra.T, aspect='auto', origin='lower',
                   extent=[0, len(spectra), freqs[0], freqs[-1]],
                   cmap='viridis')
    
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Normalized Frequency')
    ax.set_title('Power Spectrum')
    
    plt.colorbar(im, ax=ax, label='Power (dB)')
    
    return im


def plot_elevation_angle(data, ax=None, method='standard', **kwargs):
    """
    Plot elevation angle measurements
    
    Parameters:
    -----------
    data : dict or RadarData
        Data containing elevation angle information
    ax : matplotlib.Axes, optional
        Axes to plot on
    method : str
        Elevation calculation method ('standard', 'interferometer')
    **kwargs : dict
        Additional plotting arguments
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract elevation data
    if isinstance(data, dict):
        if 'elevation' in data:
            elev_data = data['elevation']
        else:
            raise ValueError("No elevation data found")
        times = data.get('time', np.arange(elev_data.shape[0]))
        ranges = data.get('range', np.arange(elev_data.shape[-1]))
    else:
        elev_data = data.elevation
        times = data.time
        ranges = data.range
    
    # Handle GPU data
    if hasattr(elev_data, 'get'):
        elev_data = elev_data.get()
    
    # Average over beams if 3D
    if elev_data.ndim == 3:
        elev_data = np.nanmean(elev_data, axis=1)
    
    # Create the plot
    im = ax.imshow(elev_data.T, aspect='auto', origin='lower',
                   cmap='coolwarm', vmin=0, vmax=60,
                   extent=[times[0], times[-1], ranges[0], ranges[-1]])
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Range Gate')
    ax.set_title(f'Elevation Angle ({method})')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Elevation Angle (degrees)')
    
    return im


def plot_grid(grid_data, parameter='velocity', ax=None, projection='geo', **kwargs):
    """
    Plot gridded SuperDARN data on geographic/magnetic coordinates
    
    Parameters:
    -----------
    grid_data : dict or GridData
        Gridded SuperDARN data
    parameter : str
        Parameter to plot
    ax : matplotlib.Axes, optional
        Axes to plot on
    projection : str
        Coordinate system ('geo', 'mag', 'mlt')
    **kwargs : dict
        Additional plotting arguments
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    # Extract grid data
    if isinstance(grid_data, dict):
        plot_data = grid_data[parameter]
        lats = grid_data.get('lat', None)
        lons = grid_data.get('lon', None)
    else:
        plot_data = getattr(grid_data, parameter)
        lats = grid_data.lat
        lons = grid_data.lon
    
    # Handle GPU data
    if hasattr(plot_data, 'get'):
        plot_data = plot_data.get()
    if hasattr(lats, 'get'):
        lats = lats.get()
    if hasattr(lons, 'get'):
        lons = lons.get()
    
    # Set color scheme
    if parameter == 'velocity':
        cmap = 'RdBu_r'
        vmin, vmax = kwargs.get('vmin', -1000), kwargs.get('vmax', 1000)
        label = 'Line-of-sight Velocity (m/s)'
    elif parameter == 'power':
        cmap = 'viridis'
        vmin, vmax = kwargs.get('vmin', 0), kwargs.get('vmax', 30)
        label = 'Backscatter Power (dB)'
    else:
        cmap = 'viridis'
        vmin, vmax = kwargs.get('vmin', None), kwargs.get('vmax', None)
        label = parameter
    
    # Create scatter plot (basic implementation)
    # In a full implementation, this would use proper map projections
    scatter = ax.scatter(lons.flatten(), lats.flatten(), 
                        c=plot_data.flatten(), cmap=cmap, 
                        vmin=vmin, vmax=vmax, s=20, **kwargs)
    
    ax.set_xlabel('Longitude (degrees)')
    ax.set_ylabel('Latitude (degrees)')
    ax.set_title(f'Gridded {parameter.title()} ({projection.upper()})')
    
    plt.colorbar(scatter, ax=ax, label=label)
    
    # Add coastlines if available
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        
        # This would be enhanced with proper cartopy integration
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
    except ImportError:
        pass
    
    return scatter


def plot_convection_map(map_data, ax=None, show_vectors=True, 
                       vector_scale=1.0, **kwargs):
    """
    Plot convection map with equipotential contours and flow vectors
    
    Parameters:
    -----------
    map_data : dict or ConvectionMap
        Convection mapping results
    ax : matplotlib.Axes, optional
        Axes to plot on
    show_vectors : bool
        Whether to show velocity vectors
    vector_scale : float
        Scale factor for velocity vectors
    **kwargs : dict
        Additional plotting arguments
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    # Extract map data
    if isinstance(map_data, dict):
        potential = map_data.get('potential', None)
        velocity_x = map_data.get('velocity_x', None)
        velocity_y = map_data.get('velocity_y', None)
        lats = map_data.get('lat', None)
        lons = map_data.get('lon', None)
    else:
        potential = map_data.potential
        velocity_x = map_data.velocity_x
        velocity_y = map_data.velocity_y
        lats = map_data.lat
        lons = map_data.lon
    
    # Handle GPU data
    for data in [potential, velocity_x, velocity_y, lats, lons]:
        if hasattr(data, 'get'):
            data = data.get()
    
    # Plot equipotential contours
    if potential is not None:
        contours = ax.contour(lons, lats, potential, levels=20, 
                            colors='black', linewidths=1.0, alpha=0.7)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%d kV')
    
    # Plot velocity vectors
    if show_vectors and velocity_x is not None and velocity_y is not None:
        # Subsample for cleaner visualization
        step = max(1, len(lons) // 20)
        ax.quiver(lons[::step, ::step], lats[::step, ::step],
                 velocity_x[::step, ::step] * vector_scale,
                 velocity_y[::step, ::step] * vector_scale,
                 angles='xy', scale_units='xy', scale=1, 
                 color='red', width=0.003)
    
    ax.set_xlabel('Longitude (degrees)')
    ax.set_ylabel('Latitude (degrees)')
    ax.set_title('Ionospheric Convection Map')
    
    # Add coordinate grid
    ax.grid(True, alpha=0.3)
    
    return ax


def create_summary_plot(data, title="SuperDARN Data Summary"):
    """
    Create comprehensive summary plot showing multiple parameters
    
    Parameters:
    -----------
    data : dict or RadarData
        SuperDARN data with multiple parameters
    title : str
        Overall plot title
        
    Returns:
    --------
    fig : matplotlib.Figure
        Summary figure with subplots
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=16)
    
    # Available parameters to plot
    params_to_plot = ['velocity', 'power', 'width', 'elevation']
    available_params = []
    
    # Check which parameters are available
    if isinstance(data, dict):
        available_params = [p for p in params_to_plot if p in data]
    else:
        available_params = [p for p in params_to_plot if hasattr(data, p)]
    
    # Plot available parameters
    for i, param in enumerate(available_params[:6]):
        row, col = i // 3, i % 3
        try:
            plot_range_time(data, parameter=param, ax=axes[row, col])
        except Exception as e:
            axes[row, col].text(0.5, 0.5, f'Error plotting {param}:\n{str(e)}',
                               ha='center', va='center', transform=axes[row, col].transAxes)
    
    # Hide unused subplots
    for i in range(len(available_params), 6):
        row, col = i // 3, i % 3
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    return fig
