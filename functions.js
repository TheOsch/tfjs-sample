const displayArrayLine = (v) => {
  if (v.length <= 10) {
    console.log('[', ... v, ']');
  } else {
    console.log('[', ... v.slice(0, 10), '...]');
  }
};

module.exports = {
  displayMatrix: (m) => {
    console.log('Size: ', m.length, ' x ', m[0].length);
    for (let i = 0; i < 5 && i < m.length; ++i) {
      displayArrayLine(m[i]);
    }
    if (m.length > 5) {
      console.log('...');
    }
  },
  displayVector: (v) => {
    console.log('Length: ', v.length);
    displayArrayLine(v);
  }
};