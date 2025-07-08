local common = import 'common.jsonnet';

{
  name: 'FY-3C, MWHS',
  filename: 'fy3c.json',
  description: "The data from the MWHS microwave radiometer onboard FY-3C, a Fengyun satellite. Data is available for three FY-3 satellites, FY-3A, FY-3B and FY-3C.",
  title: 'FY-3C, MWHS',
  unique_id: '12',
  start_datetime: '2013-09-30T21:05:22',
  end_datetime: '2018-12-31T19:58:58',
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
    dataset_name: 'MICROWAVE_FCDR_V1.1-20200512/FY3C/*/*.nc',
    type: 'application/vnd+netcdf',
    description: 'Observation dataset',
    locations: [common.hpc.hpc2020],
    size: '0.5 TB',
    inodes: '10',
    roles: ['data'],
  },
}
