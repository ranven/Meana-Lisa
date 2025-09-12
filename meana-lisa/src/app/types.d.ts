interface Artwork {
  id: number;
  primaryImage: string;
  artistDisplayName: string;
  title: string;
  medium: string;
  period: string;
  year: number; // = objectEndDate
  classification: string;
  primaryColor: RGBString;
  palette: RGBString[];
  department: string;
  isHighlight: boolean;
  culture: string;
}

interface Year {
  id: number;
  artworks: number[]; // Array of Artwork IDs
  meanPalette: RGBString[];
}

type RGBString = `rgb(${number}, ${number}, ${number})`;
