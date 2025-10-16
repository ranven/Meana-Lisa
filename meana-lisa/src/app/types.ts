type Value = {
  value: any;
  accuracy: number;
};

type Color = [string, number];

interface SimilarPainting {
  _id: string;
  objectID: number;
  isHighlight: boolean;
  primaryImage: string;
  department: string;
  objectName: string;
  title: string;
  artistDisplayName: string;
  artistNationality: string;
  artistBeginDate: number;
  artistEndDate: number;
  artistWikidata_URL: string;
  objectBeginDate: number;
  objectEndDate: number;
  medium: string;
  dimensions: string;
  classification: string;
  objectURL: string;
}

interface SimilarPalette {
  similar: SimilarPainting[];
}

interface Meana {
  department: Value;
  nat: Value;
  century: Value;
  palette: Color[];
  similar_palette: SimilarPalette;
}
