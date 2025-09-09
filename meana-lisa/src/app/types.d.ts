interface Artwork {
  id: number;
  primaryImage: string;
  objectName: string;
  title: string;
  medium: string;
  period: string;
  year: number; // = objectEndDate
  classification: string;
  primaryColor: RGBString;
  palette: RGBString[];
}

interface Year {
  id: number;
  artworks: number[]; // Array of Artwork IDs
  meanPalette: RGBString[];
}

type RGBString = `rgb(${number}, ${number}, ${number})`;
