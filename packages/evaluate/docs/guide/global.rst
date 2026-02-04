Global Plotting Options
=======================

.. code-block::json

    global_plotting_options:
      regions: ["belgium", "global"]
      image_format : "png" #options: "png", "pdf", "svg", "eps", "jpg" ..
      dpi_val : 300  # resolution of saved images in DPI
      fps: 2 #Frames Per Second, for creating a .gif file from the images; used if "plot_animations: True"
      ERA5:
        marker_size: 2
        scale_marker_size: 1
        marker: "o"
        # alpha: 0.5
        2t: 
          vmin: 250
          vmax: 300
        10u:
          vmin: -40
          vmax: 40


These control the values used to visualise the metrics, e.g. marker sizes, resolution of the images and file formats.
They will apply to all the run ids mentioned in the rest of the config.