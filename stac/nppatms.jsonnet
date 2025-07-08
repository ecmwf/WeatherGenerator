local common = import 'common.jsonnet';

{
  name: 'NPP-ATMS',
  filename: 'npp-atms.json',
  description: 'The NPP-ATMS (Advanced Technology Microwave Sounder) dataset is derived from the ATMS instrument onboard the NOAA/NASA National Polar-orbiting Partnership (NPP) satellite. It provides global measurements of atmospheric temperature, moisture, and pressure profiles, crucial for weather forecasting and climate monitoring',
  title: 'NPP-ATMS',
  unique_id: '6',
  start_datetime: '2011-12-11T00:36:13',
  end_datetime: '2018-12-31T23:58:08',
  frequency: 'NA',
  fixed_timesteps: 'False',
  keywords: [
    'atmosphere',
    'observation',
    'polar-orbiter',
    'satellite',
  ],
  providers: [
    common.providers.ecmwf_host,
    common.providers.eumetsat,
  ],
  processing_level: '1C',

  variables: {
    names: [
      'quality_pixel_bitmask',
      'instrtemp',
      'scnlin',
      'satellite_azimuth_angle',
      'satellite_zenith_angle',
      'solar_azimuth_angle',
      'solar_zenith_angle',
      'data_quality_bitmask',
      'quality_scanline_bitmask',
      'time',
      'warmnedt',
      'coldnedt',
      'btemps',
      'u_independent_btemps',
      'u_structured_btemps',
      'u_common_btemps',
      'quality_issue_pixel_bitmask',
    ],

  },

  geometry: [-180, 180, -90, 90],

  dataset: {
    dataset_name: 'MICROWAVE_FCDR_V1.1-20200512/SNPP/*/*.nc',
    type: 'application/vnd+netcdf',
    description: 'Observation dataset',
    locations: [common.hpc.hpc2020, common.hpc.jsc],
    size: '2.9 TB',
    inodes: '44469',
    roles: ['data'],
  },
}
