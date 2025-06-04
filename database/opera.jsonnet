local common = import 'common.jsonnet';

{
  name: 'OPERA',
  filename: 'opera.json',
  description: 'The OPERA radar dataset is produced by the EUMETNET OPERA program, which coordinates and harmonizes European weather radar observations. It provides quality-controlled, pan-European radar composites and individual radar data from national meteorological services. ',
  title: 'OPERA',
  unique_id: '3',
  start_datetime: '2013-01-22T15:05:00',
  end_datetime: '2024-02-15T14:05:00',
  frequency: '15m',
  fixed_timesteps: 'True',
  keywords: [
    'radar',
    'precipitation',
    'atmosphere',
    'observations',
  ],
  providers: [
    common.providers.ecmwf_host,
  ],
  variables: {
    names: ['mask', 'quality', 'tp'],
    mins: [0, 0, 0],
    maxs: [3, 24.6, 1.09959e+19],
    means: [1.24214, 233054, 2.9961e+12],
    stds: [0.646755, 0.195426, 2.78072e+15],
  },

  geometry: [-39.5, 57.7, 31.8, 73.9],

  dataset: {
    dataset_name: 'rodeo-opera-files-2km-2013-2023-15m-v1-lambert-azimuthal-equal-area.zarr',
    type: 'application/vnd+zarr',
    description: 'Anemoi dataset',
    locations: [common.hpc.hpc2020, common.hpc.jsc, common.hpc.marenostrum5, common.hpc.jsc],
    size: '959 GiB',
    inodes: '380,987',
    roles: ['data'],
  },
}
