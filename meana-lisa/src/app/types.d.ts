interface Artwork {
  // general data
  id: number;
  isHighlight: boolean;
  primaryImage: string;
  department: string;
  objectName: string;
  title: string;

  // artist data
  artistDisplayName: string;
  artistNationality: string;
  artistBeginDate: string;
  artistEndDate: string;
  artistWikiUrl: string;

  // artwork data
  objectBeginDate: string;
  objectEndDate: string;
  medium: string;
  dimensions: string;
  classification: string;
  objectUrl: string;

  // color data
  primaryColor: RGBString;
  palette: RGBString[];
}

interface Year {
  id: number;
  artworks: number[]; // Array of Artwork IDs
  meanPalette: RGBString[];
}

type RGBString = `rgb(${number}, ${number}, ${number})`;
