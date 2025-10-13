type Value = {
  value: any;
  accuracy: number;
};

type Color = [string, number];

interface Meana {
  department: Value;
  nat: Value;
  century: Value;
  palette: Color[];
}
