export interface AppState {
  productionVersion: string;
  knowledgeCoverage: number;
  retentionRate: number;
}

export const getAppState = (): AppState => {
  return {
    productionVersion: 'v2.4.1',
    knowledgeCoverage: 0.82,
    retentionRate: 0.987,
  };
};
